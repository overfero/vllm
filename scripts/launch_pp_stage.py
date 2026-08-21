#!/usr/bin/env python3
"""Official launcher for one stage of a transport-backed, multi-machine
pipeline-parallel GPT-OSS 120B deployment (see
README_RUN_GPTOSS_CLUSTER.md).

What this script does, concretely:

1. Sets the environment variables `vllm.transport.pp_worker.TransportPPWorker`
   reads (pp rank/world size, neighbor names, transport backend/ports) -
   these are inherited by every worker process vLLM spawns for local TP,
   since `update_environment_variables` (vllm/v1/executor/
   multiproc_executor.py) only overwrites the specific keys vLLM itself
   manages, never clears the environment.
2. Execs the real, completely unmodified `vllm serve` CLI
   (`python3 -m vllm.entrypoints.cli.main serve ...`) with
   `--tensor-parallel-size` set to this machine's LOCAL GPU count (real
   NCCL), `--pipeline-parallel-size 1` (deliberately - see below), and
   `--worker-cls vllm.transport.pp_worker.TransportPPWorker` so every
   local worker process installs the transport-backed PP group right
   after local bootstrap and before model load (see pp_worker.py).

Why `--pipeline-parallel-size 1`: vLLM's own `ParallelConfig.
pipeline_parallel_size` controls how many ranks its *local*
`init_distributed_environment`/`ensure_model_parallel_initialized`
bootstrap expects to rendezvous with directly - setting it to the real
value (3) would make every machine's local bootstrap try to form a
torch.distributed group spanning all 3 machines, which is exactly the
NAT-reachability problem this project's transport exists to avoid needing
at all. Layer partitioning (`make_layers()` -> `get_pp_indices()`) and
`is_first_rank`/`is_last_rank` are read from the *live* `_PP` group at
model-construction time (`vllm/model_executor/models/utils.py`), not
from this config value, so `TransportPPWorker` overwriting `_PP` with the
real 3-stage synthetic group before `load_model()` runs is what actually
determines each stage's real layer shard - confirmed correct by
`tests/transport/test20_real_bootstrap_pp_three_stage.py`.

FORMERLY A KNOWN GAP, NOW CLOSED FOR THE DRIVER STAGE: vLLM's
`EngineCore` computes `scheduler_output` exactly once per step (one
`Scheduler`) and dispatches it via `Executor.collective_rpc()`.
`MultiprocExecutor`'s own remote-reader path needs direct TCP
reachability between nodes, which this project's NAT'd machines don't
have - so `vllm/transport/rpc_executor.py`'s `TransportExecutor` forwards
each per-step call to every remote stage's `scripts/stage_server.py`
over a dedicated RPC control channel instead (separate from the PP
tensor links below). Pass `--remote-stage-names`/`--remote-stage-hosts`/
`--num-gpu-blocks-override` (this file's own flags, above) on the ONE
stage that runs `--serve` to enable it. Non-driver stages no longer run
this script at all for their engine process - they run
`scripts/stage_server.py`, which loads their shard, connects the same PP
tensor link(s), and then waits for the driver to forward it
`scheduler_output` instead of running its own (uncoordinated) Scheduler.
See `vllm/transport/rpc_executor.py` and `scripts/stage_server.py` module
docstrings for the full design, and `pp_tests/BLOCKER_REPORT.md` Blocker
1/3 for what was verified and what still needs real 3-machine hardware to
confirm end-to-end.

Usage (see README_RUN_GPTOSS_CLUSTER.md for the exact 3-machine
commands):

    python3 scripts/launch_pp_stage.py \\
        --model $MODEL_PATH \\
        --tensor-parallel-size 2 \\
        --pp-rank 0 --pp-world-size 3 \\
        --self-name MachineA --next-name MachineB \\
        --transport udp --signaling-url $SIGNALING_URL \\
        --quantization gptq --dtype float16
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def _maybe_start_quic_broker_daemons(args: argparse.Namespace) -> None:
    """Driver-side counterpart of stage_server.py's function of the same
    name - see that one's docstring for the full rationale. The role
    (`--listen` or not) assigned to each peer here must be the mirror
    image of whatever the PEER machine's own daemon uses for the SAME
    connection, since exactly one side of a QUIC connection must present
    a TLS certificate (`--listen`) and the other must not - prev-link
    peers get `listen=False` here (this driver is always the higher-rank/
    connecting side of its prev link, matching pipeline_bootstrap.py's
    own `we_listen = not is_prev` convention for every other backend),
    next-link peers get `listen=True` (mirrors the non-driver stage on
    the other end using stage_server.py, which listens toward ITS
    next-link peer - only relevant if this driver is not the last rank),
    and every `--remote-stage-names` RPC target gets `listen=False` (the
    driver always connects for the RPC control channel, matching
    rpc_executor.py's own pre-existing `listen=False`).

    Deliberately started here, BEFORE `os.execvpe()` replaces this
    process - see `quic_broker_daemon.py`'s module docstring for why a
    thread-based broker wouldn't survive that exec, and why this spawns
    a real detached subprocess instead.
    """
    if args.transport != "quic-shared":
        return
    daemon_module = "vllm.transport.quic_broker_daemon"

    import subprocess
    import tempfile

    remote_names = (
        [n.strip() for n in args.remote_stage_names.split(",") if n.strip()]
        if args.remote_stage_names else []
    )
    peers = []
    for name, listen in ((args.prev_name, False), (args.next_name, True)):
        if name and name not in [p for p, _ in peers]:
            peers.append((name, listen))
    for name in remote_names:
        if name not in [p for p, _ in peers]:
            peers.append((name, False))

    for i, (peer_name, listen) in enumerate(peers):
        ready_fd, ready_path = tempfile.mkstemp(prefix="quic_broker_ready_")
        os.close(ready_fd)
        os.remove(ready_path)
        cmd = [
            sys.executable, "-m", daemon_module,
            "--self-id", args.self_name, "--peer-id", peer_name,
            "--signaling-url", args.signaling_url,
            "--udp-port", str(args.udp_port_base + i),
            "--connect-timeout", str(args.transport_connect_timeout),
            "--ready-file", ready_path,
        ]
        if listen:
            cmd.append("--listen")
        print(f"[launch_pp_stage] starting quic_broker_daemon for peer {peer_name!r}: "
              f"{' '.join(cmd)}", file=sys.stderr, flush=True)
        proc = subprocess.Popen(cmd, start_new_session=True)

        deadline = time.monotonic() + args.transport_connect_timeout + 30.0
        while not os.path.exists(ready_path):
            if proc.poll() is not None:
                raise RuntimeError(
                    f"quic_broker_daemon for peer {peer_name!r} exited early "
                    f"(code={proc.returncode}) before becoming ready"
                )
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"quic_broker_daemon for peer {peer_name!r} did not become "
                    f"ready within {args.transport_connect_timeout + 30.0}s"
                )
            time.sleep(0.2)
        print(f"[launch_pp_stage] quic_broker_daemon for peer {peer_name!r} ready",
              file=sys.stderr, flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # --- pipeline topology (this project's transport, not vLLM's own PP) ---
    p.add_argument("--pp-rank", type=int, required=True, help="0-indexed stage rank of this machine")
    p.add_argument("--pp-world-size", type=int, required=True, help="total pipeline stages (3 for the target cluster)")
    p.add_argument("--self-name", required=True, help="this machine's identity, e.g. MachineA")
    p.add_argument("--prev-name", default=None, help="previous stage's machine name (omit for pp-rank 0)")
    p.add_argument("--next-name", default=None, help="next stage's machine name (omit for the last pp-rank)")

    # --- transport (vllm/transport/ - frozen, unmodified) ---
    p.add_argument("--transport", choices=["tcp", "udp", "quic", "quic-shared"], default="udp")
    p.add_argument("--signaling-url", default=None, help="required for --transport udp")
    p.add_argument("--udp-port-base", type=int, default=30000)
    p.add_argument("--tcp-port-base", type=int, default=30000)
    p.add_argument("--tcp-connect-host-prev", default=None, help="--transport tcp only: prev stage's reachable address")
    p.add_argument("--tcp-connect-host-next", default=None, help="--transport tcp only: next stage's reachable address")
    p.add_argument("--transport-connect-timeout", type=float, default=120.0)

    # --- model / vLLM passthrough (only the flags this deployment needs) ---
    p.add_argument("--model", required=True, help="local path or HF repo id of this stage's checkpoint")
    p.add_argument("--tensor-parallel-size", type=int, default=2, help="local GPU count on this machine")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--quantization", default=None, choices=[None, "awq", "gptq", "awq_marlin", "gptq_marlin"])
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--kv-cache-dtype", default=None,
                    help="Real vLLM CLI flag (e.g. int8_per_token_head) - must match "
                         "every scripts/stage_server.py's own --kv-cache-dtype too.")
    p.add_argument("--max-num-batched-tokens", type=int, default=None,
                    help="Real vLLM CLI flag - must match every scripts/stage_server.py's "
                         "own --max-num-batched-tokens too (same reasoning as "
                         "--speculative-config: the driver dispatches per-step batches "
                         "sized against this value).")
    p.add_argument("--enable-auto-tool-choice", action="store_true",
                    help="Real vLLM CLI flag, driver/API-server-only - lets the model "
                         "choose whether to call a tool vs answer directly.")
    p.add_argument("--tool-call-parser", default=None,
                    help="Real vLLM CLI flag, driver/API-server-only - required "
                         "alongside --enable-auto-tool-choice for tool-call output to "
                         "parse correctly (e.g. qwen3_xml for this model family).")
    p.add_argument("--reasoning-parser", default=None,
                    help="Real vLLM CLI flag, driver/API-server-only - separates "
                         "<think>...</think> reasoning into its own response field "
                         "instead of leaking it into visible text (e.g. qwen3).")
    p.add_argument("--api-key", default=None,
                    help="Real vLLM CLI flag, driver/API-server-only - requires "
                         "'Authorization: Bearer <key>' on every request when set.")
    p.add_argument("--max-num-seqs", type=int, default=None,
                    help="see stage_server.py's matching flag docstring - required <= "
                         "--num-gpu-blocks-override for hybrid Mamba models under CUDA graph")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--language-model-only", action="store_true",
                    help="skip the vision tower for multimodal models served text-only")
    p.add_argument("--enable-cudagraph", action="store_true",
                    help="EXPERIMENTAL - see stage_server.py's matching flag docstring. "
                         "Must be passed on every stage + the driver together.")
    p.add_argument("--speculative-config", default=None,
                    help="Real, existing vLLM flag (JSON string), passed through as-is - "
                         "e.g. '{\"method\": \"mtp\", \"num_speculative_tokens\": 1}' for "
                         "Qwen3.5's native MTP draft head. Only meaningful on the DRIVER "
                         "(this stage/--serve): the MTP module runs entirely on the last "
                         "PP stage since it consumes the target model's own final "
                         "hidden_states directly, no cross-machine transport of its own - "
                         "see vllm/model_executor/models/qwen3_5_mtp.py's "
                         "get_pp_group().is_last_rank checks. The checkpoint at --model "
                         "must have been extracted with --include-mtp.")
    p.add_argument("--enable-expert-parallel", action="store_true",
                    help="PP3+EP2 candidate (2026-08-15 benchmark) - see stage_server.py's "
                         "matching flag docstring. Must be passed on every stage + the "
                         "driver together (each stage's local TP group independently "
                         "becomes its EP group; the driver has no special role here).")
    p.add_argument("--enable-pipelining", action="store_true",
                    help="Microbatching experiment (2026-08-15) - see "
                         "vllm/transport/rpc_executor.py's module docstring and "
                         "stage_server.py's matching flag. Must be passed on every stage "
                         "+ the driver together, never partially.")
    p.add_argument("--batch-queue-size", type=int, default=None,
                    help="Microbatching experiment (2026-08-15), driver-only (this "
                         "machine's TransportExecutor is the only one running a real, "
                         "driven Scheduler - see vllm/transport/rpc_executor.py's module "
                         "docstring for exactly what this mutates and why it's safe for "
                         "already-spawned local workers). Requires --enable-pipelining "
                         "too (enforced at runtime in rpc_executor.py, not here). "
                         "Typically the real cross-machine stage count (--pp-world-size).")
    p.add_argument("--enable-rpc-fusion", action="store_true",
                    help="RPC fusion (2026-08-15), on top of --enable-pipelining: "
                         "combines a step's execute_model+sample_tokens into ONE RPC "
                         "round trip per remote link instead of two (see "
                         "rpc_executor.py's _forward_and_local_fused). Must be passed "
                         "on every stage + the driver together (needs "
                         "--enable-pipelining on all of them too).")

    # --- API server (only meaningful on the stage you expose to clients) ---
    p.add_argument("--skip-mm-profiling", action="store_true",
                    help="Real vLLM CLI flag - skip the multimodal encoder/encoder-cache "
                         "memory-profiling dummy run at startup. Needed on a driver that "
                         "accepts image/video input at the API layer (no "
                         "--language-model-only) but isn't the first PP rank: this "
                         "profiling pass calls the model's embed_multimodal() with a "
                         "dummy batch regardless of PP rank, which crashes here since "
                         "the vision tower was never constructed locally on a non-first "
                         "rank (see qwen3_5.py's PP-rank-aware tower skip). Safe on this "
                         "topology because the driver never runs the vision encoder "
                         "itself anyway - only the first PP rank ever sees raw "
                         "pixel_values, every later rank only receives forwarded hidden "
                         "states, so there's no real encoder cache for this stage to "
                         "size correctly in the first place.")
    p.add_argument("--serve", action="store_true", help="expose the OpenAI-compatible API on this machine")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)

    # --- cross-machine scheduler_output RPC (vllm/transport/rpc_executor.py) ---
    # Only meaningful on the DRIVER stage (the one running the real
    # Scheduler - normally the same machine as --serve). Non-driver stages
    # do not use this script at all anymore for their engine process - see
    # scripts/stage_server.py, which they run instead. Closes
    # README_RUN_GPTOSS_CLUSTER.md Task 10's "known gap".
    p.add_argument("--remote-stage-names", default=None,
                    help="comma-separated stage_server names this stage's Scheduler must "
                         "drive every step, e.g. MachineA,MachineB (driver stage only)")
    p.add_argument("--remote-stage-hosts", default=None,
                    help="comma-separated, 1:1 with --remote-stage-names - reachable host "
                         "for each (--transport tcp only)")
    p.add_argument("--rpc-port", type=int, default=40000,
                    help="port every scripts/stage_server.py listens on for the driver's "
                         "RPC control channel (shared across stages - only the host differs)")
    p.add_argument("--num-gpu-blocks-override", type=int, default=None,
                    help="REQUIRED (and must be identical across all 3 machines) whenever "
                         "--remote-stage-names is set - see rpc_executor.py module docstring "
                         "and pp_tests/BLOCKER_REPORT.md Blocker 3 for why")
    p.add_argument("--cpu-offload-gb", type=float, default=0.0,
                    help="Real vLLM flag: offload this many GiB of model weights to pinned "
                         "host RAM instead of GPU VRAM - see stage_server.py's matching flag "
                         "docstring for why this stage's real footprint (weights + the "
                         "always-constructed-on-every-rank MTP drafter + CUDA graph capture "
                         "buffers, if --enable-cudagraph) may not fit a T4's 14.56GiB without it.")

    p.add_argument("--dry-run", action="store_true", help="print the resulting env vars and vllm command, do not exec")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    if (args.prev_name is None) != (args.pp_rank == 0):
        print("error: --prev-name must be omitted iff --pp-rank is 0", file=sys.stderr)
        return 2
    if (args.next_name is None) != (args.pp_rank == args.pp_world_size - 1):
        print("error: --next-name must be omitted iff --pp-rank is the last stage", file=sys.stderr)
        return 2
    if args.transport in ("udp", "quic", "quic-shared") and not args.signaling_url:
        print(f"error: --signaling-url is required for --transport {args.transport}", file=sys.stderr)
        return 2
    if args.remote_stage_names and args.num_gpu_blocks_override is None:
        print("error: --num-gpu-blocks-override is required when --remote-stage-names is "
              "set (must match the value passed to every scripts/stage_server.py too)",
              file=sys.stderr)
        return 2

    env = os.environ.copy()
    # so sitecustomize.py (in its own dedicated _pysitecustomize/ dir, NOT
    # the project root - see that file's docstring for why) auto-applies
    # the real SM75 humming-kernels runtime patches inside the `vllm serve`
    # subprocess this execs into.
    sitecustomize_dir = str(Path(__file__).resolve().parents[1] / "_pysitecustomize")
    env["PYTHONPATH"] = sitecustomize_dir + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["VLLM_TRANSPORT"] = args.transport
    env["VLLM_TRANSPORT_PP_RANK"] = str(args.pp_rank)
    env["VLLM_TRANSPORT_PP_WORLD_SIZE"] = str(args.pp_world_size)
    env["VLLM_TRANSPORT_SELF_NAME"] = args.self_name
    env["VLLM_TRANSPORT_PREV_NAME"] = args.prev_name or ""
    env["VLLM_TRANSPORT_NEXT_NAME"] = args.next_name or ""
    env["VLLM_TRANSPORT_SIGNALING_URL"] = args.signaling_url or ""
    env["VLLM_TRANSPORT_UDP_PORT_BASE"] = str(args.udp_port_base)
    env["VLLM_TRANSPORT_TCP_PORT_BASE"] = str(args.tcp_port_base)
    env["VLLM_TRANSPORT_TCP_CONNECT_HOST_PREV"] = args.tcp_connect_host_prev or ""
    env["VLLM_TRANSPORT_TCP_CONNECT_HOST_NEXT"] = args.tcp_connect_host_next or ""
    env["VLLM_TRANSPORT_CONNECT_TIMEOUT"] = str(args.transport_connect_timeout)
    if args.enable_pipelining:
        env["VLLM_TRANSPORT_ENABLE_PIPELINING"] = "1"
    if args.batch_queue_size is not None:
        env["VLLM_TRANSPORT_BATCH_QUEUE_SIZE"] = str(args.batch_queue_size)
    if args.enable_rpc_fusion:
        env["VLLM_TRANSPORT_ENABLE_RPC_FUSION"] = "1"
    if args.remote_stage_names:
        env["VLLM_TRANSPORT_REMOTE_STAGE_NAMES"] = args.remote_stage_names
        env["VLLM_TRANSPORT_REMOTE_STAGE_HOSTS"] = args.remote_stage_hosts or ""
        env["VLLM_TRANSPORT_RPC_PORT"] = str(args.rpc_port)

    executor_backend = (
        "vllm.transport.rpc_executor.TransportExecutor"
        if args.remote_stage_names
        else "mp"
    )
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve",
        args.model,
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--pipeline-parallel-size", "1",  # local-only; see module docstring
        "--distributed-executor-backend", executor_backend,
        "--worker-cls", "vllm.transport.pp_worker.TransportPPWorker",
        "--dtype", args.dtype,
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--host", args.host,
        "--port", str(args.port),
        # Real bug hit running this for real: with async scheduling on (vLLM's
        # default whenever the executor supports it), every non-last PP rank
        # calls _pp_receive_prev_sampled_token_ids_to_input_batch(), which does
        # a real `torch.distributed.broadcast(..., group=pp.device_group)` to
        # pull the last rank's sampled token ids directly - a second, separate
        # cross-rank channel from the activation-tensor transport this project
        # implements, and one the synthetic transport-backed PP group (built
        # via object.__new__(), see pipeline_bootstrap.py) has no real
        # multi-machine backing for at all
        # (`AttributeError: 'GroupCoordinator' object has no attribute
        # 'device_group'`). Must disable async scheduling on every stage.
        "--no-async-scheduling",
    ]
    if args.quantization:
        cmd += ["--quantization", args.quantization]
    if args.max_model_len:
        cmd += ["--max-model-len", str(args.max_model_len)]
    if args.max_num_seqs:
        cmd += ["--max-num-seqs", str(args.max_num_seqs)]
    if args.speculative_config:
        cmd += ["--speculative-config", args.speculative_config]
    if args.trust_remote_code:
        cmd += ["--trust-remote-code"]
    if args.language_model_only:
        cmd += ["--language-model-only"]
    if not args.enable_cudagraph:
        # Default: every prior real deployment on this project forced eager
        # execution on every stage (no CUDA graphs, no compile). Real bug hit
        # running this for real: with only ONE stage in compiled/CUDA-graph
        # mode, its local model runner pads intermediate-tensor shapes to the
        # nearest cudagraph capture size (e.g. 8 for a 5-token batch) before
        # gathering the activations the previous stage sent over the real PP
        # transport link - but the previous stage sent the real, unpadded
        # size (5), since it was eager. Plain shape mismatch
        # (`RuntimeError: size of tensor a (8) must match tensor b (5)`), not
        # a transport bug. --enable-cudagraph tests the untested hypothesis
        # that padding agrees stage-to-stage when EVERY stage uses graphs
        # consistently (see stage_server.py's matching flag docstring) - it
        # must be passed on every stage + the driver together, never
        # partially, or this exact crash reproduces.
        cmd += ["--enforce-eager"]
    if args.num_gpu_blocks_override is not None:
        cmd += ["--num-gpu-blocks-override", str(args.num_gpu_blocks_override)]
    if args.cpu_offload_gb:
        cmd += ["--cpu-offload-gb", str(args.cpu_offload_gb)]
    if args.enable_expert_parallel:
        cmd += ["--enable-expert-parallel"]
    if args.kv_cache_dtype:
        cmd += ["--kv-cache-dtype", args.kv_cache_dtype]
    if args.max_num_batched_tokens:
        cmd += ["--max-num-batched-tokens", str(args.max_num_batched_tokens)]
    if args.enable_auto_tool_choice:
        cmd += ["--enable-auto-tool-choice"]
    if args.tool_call_parser:
        cmd += ["--tool-call-parser", args.tool_call_parser]
    if args.reasoning_parser:
        cmd += ["--reasoning-parser", args.reasoning_parser]
    if args.api_key:
        cmd += ["--api-key", args.api_key]
    if args.skip_mm_profiling:
        cmd += ["--skip-mm-profiling"]
    if not args.serve:
        # Non-serving stages still need a real engine constructed (to load
        # their shard and connect their transport link(s)) - see the
        # module docstring's "known gap" note for what "running" means
        # for these stages today. `--api-server-count 0` isn't a real
        # vLLM flag; there is currently no supported way to construct the
        # engine without also starting an (unused, harmless) local HTTP
        # server, so every stage gets one, bound to localhost only unless
        # --serve was passed.
        cmd[cmd.index("--host") + 1] = "127.0.0.1"

    print(f"[launch_pp_stage] pp_rank={args.pp_rank}/{args.pp_world_size} "
          f"self={args.self_name} prev={args.prev_name} next={args.next_name} "
          f"transport={args.transport}", file=sys.stderr)
    print(f"[launch_pp_stage] exec: {' '.join(cmd)}", file=sys.stderr)

    if args.dry_run:
        print("[launch_pp_stage] --dry-run: not executing", file=sys.stderr)
        return 0

    _maybe_start_quic_broker_daemons(args)

    os.execvpe(cmd[0], cmd, env)
    return 0  # unreachable - execvpe replaces this process on success


if __name__ == "__main__":
    sys.exit(main())
