#!/usr/bin/env python3
"""Official launcher for a NON-DRIVER stage in a transport-backed,
multi-machine pipeline-parallel GPT-OSS 120B deployment (see
README_RUN_GPTOSS_CLUSTER.md and vllm/transport/rpc_executor.py).

Replaces running `scripts/launch_pp_stage.py` (without `--serve`) for
every stage except the one running the real Scheduler (the "driver" -
normally the same machine as `--serve`, e.g. Machine C). The old
approach gave every stage its own independent `vllm serve` process, each
with its own uncoordinated `Scheduler` - `EngineCore.step()` on a
non-driver stage never fired for the same request the driver was
handling, so `get_pp_group().irecv_tensor_dict()` on that stage's workers
blocked forever (README_RUN_GPTOSS_CLUSTER.md Task 10).

This script instead:

1. Sets the same `VLLM_TRANSPORT_*` env vars `TransportPPWorker` reads
   (identical to `launch_pp_stage.py` - the PP activation-tensor links
   this establishes are completely unchanged/unaffected by anything in
   this file).
2. Builds a real `vllm.v1.engine.core.EngineCore` directly (NOT via
   `vllm serve`'s CLI/API-server path) with a plain, unmodified
   `MultiprocExecutor` (this stage is a leaf - it never forwards to
   anyone else, so it needs none of `TransportExecutor`'s remote-dispatch
   logic). Constructing `EngineCore` does real, local-only work: load
   this stage's checkpoint shard, profile this GPU's available memory,
   build this stage's own KV cache config, warm up/compile. All of that
   is correctly stage-local and is NOT something the driver should
   compute on this stage's behalf (different layers, different memory
   pressure) - only `num_gpu_blocks` must agree across stages, which is
   why `--num-gpu-blocks-override` is a required argument here, not
   optional (see `vllm/transport/rpc_executor.py` module docstring).
   `EngineCore.__init__` also constructs a real `Scheduler` as a side
   effect - built but never driven; harmless, and simpler/more correct
   than reimplementing `_initialize_kv_caches` by hand.
3. Opens ONE listening RPC control-channel `Transport` connection for the
   driver (`--driver-name`), then loops: receive a
   `(method, args, kwargs)` tuple, call the matching method directly on
   `engine_core.model_executor` (the real local `MultiprocExecutor` -
   `execute_model`/`sample_tokens`/`execute_dummy_batch`), send back
   `(status, result_or_error)`. No new serialization format - this is the
   same `cloudpickle` wire format `TransportExecutor` sends and
   `MultiprocExecutor`'s own `collective_rpc` already uses internally.

Usage (see README_RUN_GPTOSS_CLUSTER.md for the exact 3-machine
commands; example for Machine A, stage 0 of 3, driven by Machine C):

    python3 scripts/stage_server.py \\
        --model $MODEL_PATH \\
        --tensor-parallel-size 2 \\
        --pp-rank 0 --pp-world-size 3 \\
        --self-name MachineA --next-name MachineB \\
        --driver-name MachineC \\
        --transport udp --signaling-url $SIGNALING_URL \\
        --quantization gptq --dtype float16 \\
        --num-gpu-blocks-override 4096
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import importlib.util
from pathlib import Path

# Load humming_fix/patch.py by direct file path (NOT via sys.path + normal
# package import) - the project root also contains a directory literally
# named `vllm` (this checkout), so adding the project root to sys.path for
# `import humming_fix.patch` created a namespace-package collision with the
# real editable-installed `vllm` package (real bug hit running this for
# real: `ImportError: cannot import name 'SamplingParams' from 'vllm'
# (unknown location)` - "unknown location" is the tell of a broken merged
# namespace package). patch.py itself only imports from the real `humming`
# package + stdlib (no relative imports), so direct file-path loading is
# safe and sidesteps sys.path entirely.
_patch_path = Path(__file__).resolve().parents[1] / "humming_fix" / "patch.py"
_spec = importlib.util.spec_from_file_location("humming_fix_patch", _patch_path)
_patch_module = importlib.util.module_from_spec(_spec)
sys.modules["humming_fix_patch"] = _patch_module
_spec.loader.exec_module(_patch_module)  # both real SM75 fixes, before any HummingKernel/model use

# Propagate PYTHONPATH to include _pysitecustomize/ *before* any CUDA/
# multiprocessing init below. This process's own already-imported modules
# won't benefit (site/sitecustomize only runs once at interpreter startup,
# already past by the time we get here) - that's what the direct
# humming_fix.patch load above is for. But MultiprocExecutor's TP workers
# are separate `spawn`-started interpreters (CUDA forces spawn, see
# `_maybe_force_spawn`), so each one re-reads PYTHONPATH from this
# process's os.environ at its own startup and will auto-import
# sitecustomize.py there - the only way qwen35_mtp_pp_fix.py's MTP/PP
# compat patch (see that module's docstring) reaches worker processes,
# where the MTP drafter's dummy_run/forward actually execute.
_sitecustomize_dir = str(Path(__file__).resolve().parents[1] / "_pysitecustomize")
_existing_pythonpath = os.environ.get("PYTHONPATH", "")
if _sitecustomize_dir not in _existing_pythonpath.split(os.pathsep):
    os.environ["PYTHONPATH"] = (
        _sitecustomize_dir + (os.pathsep + _existing_pythonpath if _existing_pythonpath else "")
    )

import torch  # noqa: E402

torch.zeros(1, device="cuda")
torch.cuda.synchronize()

import cloudpickle  # noqa: E402

_STATUS_OK = "ok"
_STATUS_ERROR = "error"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # --- pipeline topology (PP tensor links - identical to launch_pp_stage.py) ---
    p.add_argument("--pp-rank", type=int, required=True)
    p.add_argument("--pp-world-size", type=int, required=True)
    p.add_argument("--self-name", required=True)
    p.add_argument("--prev-name", default=None)
    p.add_argument("--next-name", default=None)

    p.add_argument("--transport", choices=["tcp", "udp"], default="udp")
    p.add_argument("--signaling-url", default=None)
    p.add_argument("--udp-port-base", type=int, default=30000)
    p.add_argument("--tcp-port-base", type=int, default=30000)
    p.add_argument("--tcp-connect-host-prev", default=None)
    p.add_argument("--tcp-connect-host-next", default=None)
    p.add_argument("--transport-connect-timeout", type=float, default=120.0)

    # --- RPC control channel to the driver (vllm/transport/rpc_executor.py) ---
    p.add_argument("--driver-name", required=True, help="the driver stage's --self-name")
    p.add_argument("--rpc-port", type=int, default=40000)
    p.add_argument("--rpc-listen-host", default="0.0.0.0")

    # --- model / vLLM config (must match the driver's for this stage's shard) ---
    p.add_argument("--model", required=True)
    p.add_argument("--tensor-parallel-size", type=int, default=2)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--quantization", default=None, choices=[None, "awq", "gptq", "awq_marlin", "gptq_marlin"])
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--num-gpu-blocks-override", type=int, required=True,
                    help="MUST match the value passed to the driver's launch_pp_stage.py "
                         "and every other stage_server.py - see rpc_executor.py")
    p.add_argument("--max-num-seqs", type=int, default=None,
                    help="For hybrid Mamba/attention models (e.g. Qwen3.5): each decode "
                         "sequence needs its own dedicated Mamba cache block, so this MUST "
                         "be <= --num-gpu-blocks-override or CUDA graph capture raises "
                         "'max_num_seqs exceeds available Mamba cache blocks' (real error "
                         "hit with the vLLM default of 128 against a 60-block override).")
    p.add_argument("--language-model-only", action="store_true",
                    help="skip the vision tower for multimodal models served text-only "
                         "(e.g. Qwen3.5's ForConditionalGeneration checkpoints with the "
                         "vision weights stripped out of the stage shard)")
    p.add_argument("--speculative-config", default=None,
                    help="Real, existing vLLM flag (JSON string) - MUST be passed here "
                         "too (identically to the driver's launch_pp_stage.py) even though "
                         "this stage never builds the actual MTP drafter (that's gated on "
                         "get_pp_group().is_last_rank in gpu_model_runner.py, False here). "
                         "Real bug hit running this for real: without it, this stage's "
                         "EngineArgs has speculative_config=None while the driver's scheduler "
                         "produces per-step batches sized for num_speculative_tokens+1 draft "
                         "candidates, and this stage's own array pre-allocation (sized for "
                         "1 token/seq) can't broadcast the larger arrays the driver sends - "
                         "'could not broadcast input array from shape (2,) into shape (1,)'.")
    p.add_argument("--cpu-offload-gb", type=float, default=0.0,
                    help="Real vLLM flag: offload this many GiB of model weights to "
                         "pinned host RAM instead of GPU VRAM. Needed for an asymmetric "
                         "PP split where a stage's real layer count doesn't fit a T4's "
                         "14.56GiB alongside the (always-constructed-on-every-rank, see "
                         "--speculative-config's own note above) MTP drafter's extra "
                         "vocab embedding + decoder layer.")
    p.add_argument("--enable-cudagraph", action="store_true",
                    help="EXPERIMENTAL (2026-08-12 latency investigation): try running "
                         "with CUDA graphs instead of the eager mode every prior real "
                         "deployment on this project used. enforce_eager=True was forced "
                         "after a real bug: with only ONE stage in graph mode, its model "
                         "runner pads intermediate-tensor shapes to the nearest capture "
                         "size before sending them over the PP transport, while an eager "
                         "neighbor expects the unpadded size -> shape mismatch crash. "
                         "Untested hypothesis: if EVERY stage uses the same capture sizes "
                         "consistently, padding should agree stage-to-stage since all PP "
                         "ranks process the same per-step token count. Must be passed on "
                         "every stage + the driver together, never partially.")
    p.add_argument("--enable-expert-parallel", action="store_true",
                    help="PP3+EP2 candidate (2026-08-15 benchmark): reuse this stage's "
                         "local tensor-parallel group (--tensor-parallel-size, real "
                         "torch.distributed, intra-machine only) to shard MoE routed "
                         "experts whole-expert-per-rank instead of splitting every "
                         "expert's matrices in half. Does not touch the cross-machine "
                         "PP transport at all - vLLM derives the EP group from the same "
                         "TP group this stage already initializes locally, see "
                         "vllm/distributed/parallel_state.py's initialize_model_parallel(). "
                         "Requires num_experts % tensor_parallel_size == 0 (256 % 2 == 0 "
                         "for this model - safe). Off by default; PP3+TP2 remains the "
                         "baseline in cluster/qwen35_122ba10b_3machine.sh.")

    return p


def _validate(args: argparse.Namespace) -> str | None:
    if (args.prev_name is None) != (args.pp_rank == 0):
        return "--prev-name must be omitted iff --pp-rank is 0"
    if (args.next_name is None) != (args.pp_rank == args.pp_world_size - 1):
        return "--next-name must be omitted iff --pp-rank is the last stage"
    if args.transport == "udp" and not args.signaling_url:
        return "--signaling-url is required for --transport udp"
    return None


def _set_transport_env(args: argparse.Namespace) -> None:
    os.environ["VLLM_TRANSPORT"] = args.transport
    os.environ["VLLM_TRANSPORT_PP_RANK"] = str(args.pp_rank)
    os.environ["VLLM_TRANSPORT_PP_WORLD_SIZE"] = str(args.pp_world_size)
    os.environ["VLLM_TRANSPORT_SELF_NAME"] = args.self_name
    os.environ["VLLM_TRANSPORT_PREV_NAME"] = args.prev_name or ""
    os.environ["VLLM_TRANSPORT_NEXT_NAME"] = args.next_name or ""
    os.environ["VLLM_TRANSPORT_SIGNALING_URL"] = args.signaling_url or ""
    os.environ["VLLM_TRANSPORT_UDP_PORT_BASE"] = str(args.udp_port_base)
    os.environ["VLLM_TRANSPORT_TCP_PORT_BASE"] = str(args.tcp_port_base)
    os.environ["VLLM_TRANSPORT_TCP_CONNECT_HOST_PREV"] = args.tcp_connect_host_prev or ""
    os.environ["VLLM_TRANSPORT_TCP_CONNECT_HOST_NEXT"] = args.tcp_connect_host_next or ""
    os.environ["VLLM_TRANSPORT_CONNECT_TIMEOUT"] = str(args.transport_connect_timeout)
    # Deliberately NOT setting VLLM_TRANSPORT_REMOTE_STAGE_NAMES - this
    # process's own executor is a plain MultiprocExecutor (see module
    # docstring point 2), not a TransportExecutor; it never forwards.


def _build_engine_core(args: argparse.Namespace):
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    speculative_config = (
        json.loads(args.speculative_config) if args.speculative_config else None
    )

    engine_args = EngineArgs(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=1,  # local-only; see launch_pp_stage.py's docstring
        distributed_executor_backend="mp",
        worker_cls="vllm.transport.pp_worker.TransportPPWorker",
        dtype=args.dtype,
        quantization=args.quantization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        num_gpu_blocks_override=args.num_gpu_blocks_override,
        cpu_offload_gb=args.cpu_offload_gb,
        language_model_only=args.language_model_only,
        speculative_config=speculative_config,
        enforce_eager=not args.enable_cudagraph,
        enable_expert_parallel=args.enable_expert_parallel,
        # Disable vLLM's default async scheduling: it makes every non-last PP
        # rank call torch.distributed.broadcast(group=pp.device_group) to pull
        # sampled token ids directly, a second cross-rank channel the synthetic
        # transport-backed PP group (pipeline_bootstrap.py) has no real
        # multi-machine backing for - see launch_pp_stage.py's matching flag.
        async_scheduling=False,
    )
    vllm_config = engine_args.create_engine_config()
    return EngineCore(vllm_config=vllm_config, executor_class=MultiprocExecutor, log_stats=False)


def _open_driver_rpc_link(args: argparse.Namespace):
    from vllm.transport import TransportConfig, get_transport

    self_id = f"{args.self_name}-rpc-to-{args.driver_name}"
    peer_id = f"{args.driver_name}-rpc-to-{args.self_name}"
    transport = get_transport(args.transport)
    transport.connect(
        TransportConfig(
            self_id=self_id,
            peer_id=peer_id,
            host=args.rpc_listen_host,
            port=args.rpc_port,
            listen=True,  # driver connects; every stage_server listens
            signaling_url=args.signaling_url or "http://127.0.0.1:8000",
            udp_mode="preserve",
            udp_port=args.rpc_port,
            connect_timeout=args.transport_connect_timeout,
        )
    )
    return transport


def _serve_rpc_loop(transport, model_executor) -> None:
    print(f"[stage_server] ready - waiting for driver RPC calls on {transport}", file=sys.stderr, flush=True)
    while True:
        try:
            raw = transport.recv(timeout=None)
        except (TimeoutError, ConnectionError, OSError) as e:
            print(f"[stage_server] transport closed/errored: {e}", file=sys.stderr, flush=True)
            return
        method, call_args, call_kwargs = pickle.loads(raw)
        debug_timing = os.environ.get("VLLM_TRANSPORT_DEBUG_TIMING")
        t0 = time.perf_counter() if debug_timing else 0.0
        try:
            fn = getattr(model_executor, method)
            result = fn(*call_args, **call_kwargs)
            status, payload = _STATUS_OK, result
        except Exception as e:  # noqa: BLE001 - report to driver, don't crash the stage
            print(f"[stage_server] method {method!r} raised: {e!r}", file=sys.stderr, flush=True)
            status, payload = _STATUS_ERROR, repr(e)
        if debug_timing:
            print(f"[stage-exec-timing] {method}: {1000*(time.perf_counter()-t0):.1f}ms",
                  flush=True)
        transport.send(cloudpickle.dumps((status, payload), protocol=pickle.HIGHEST_PROTOCOL))


def main() -> int:
    args = build_arg_parser().parse_args()
    err = _validate(args)
    if err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    _set_transport_env(args)

    print(f"[stage_server] pp_rank={args.pp_rank}/{args.pp_world_size} self={args.self_name} "
          f"prev={args.prev_name} next={args.next_name} driver={args.driver_name} "
          f"transport={args.transport} num_gpu_blocks_override={args.num_gpu_blocks_override}",
          file=sys.stderr)

    print("[stage_server] building EngineCore (load_model + kv cache init + PP transport "
          "connect happen here)...", file=sys.stderr, flush=True)
    engine_core = _build_engine_core(args)
    print("[stage_server] EngineCore ready - model loaded, PP transport link(s) connected",
          file=sys.stderr, flush=True)

    driver_link = _open_driver_rpc_link(args)
    print(f"[stage_server] RPC control channel to driver {args.driver_name!r} connected",
          file=sys.stderr, flush=True)

    try:
        _serve_rpc_loop(driver_link, engine_core.model_executor)
    finally:
        driver_link.close()
        engine_core.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
