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

KNOWN GAP - read before relying on this for online serving: vLLM's
`EngineCore` computes `scheduler_output` exactly once per step (one
`Scheduler`, `vllm/v1/engine/core.py`'s `step()`) and dispatches it to
every worker via `Executor.collective_rpc()`. `MultiprocExecutor`'s RPC
channel (`vllm/distributed/device_communicators/shm_broadcast.py`'s
`MessageQueue`) supports a real network-capable "remote reader" path for
vanilla multi-node deployments, but it connects over plain TCP
(`connect_ip`), which assumes direct reachability between nodes - the
exact same NAT problem this project's transport was built to solve for
the PP tensor path specifically, not yet solved for this RPC/scheduling
path. Concretely: this script correctly bootstraps local TP, installs
the transport PP group, loads this stage's model shard, and connects
this stage's transport link(s) to its neighbor(s) - all of that is real
and covered by test18/test20. What it does NOT yet do is guarantee that
a non-driver machine's local `EngineCore.step()` loop ever gets invoked
with a `scheduler_output` consistent with what the request-serving
machine decided, because nothing in this repository yet relays
`scheduler_output` (or bridges `collective_rpc`) across machines. See
README_RUN_GPTOSS_CLUSTER.md's "Known gap" section for the precise
proposed fix and why it isn't implemented here.

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


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # --- pipeline topology (this project's transport, not vLLM's own PP) ---
    p.add_argument("--pp-rank", type=int, required=True, help="0-indexed stage rank of this machine")
    p.add_argument("--pp-world-size", type=int, required=True, help="total pipeline stages (3 for the target cluster)")
    p.add_argument("--self-name", required=True, help="this machine's identity, e.g. MachineA")
    p.add_argument("--prev-name", default=None, help="previous stage's machine name (omit for pp-rank 0)")
    p.add_argument("--next-name", default=None, help="next stage's machine name (omit for the last pp-rank)")

    # --- transport (vllm/transport/ - frozen, unmodified) ---
    p.add_argument("--transport", choices=["tcp", "udp"], default="udp")
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
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--trust-remote-code", action="store_true")

    # --- API server (only meaningful on the stage you expose to clients) ---
    p.add_argument("--serve", action="store_true", help="expose the OpenAI-compatible API on this machine")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)

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
    if args.transport == "udp" and not args.signaling_url:
        print("error: --signaling-url is required for --transport udp", file=sys.stderr)
        return 2

    env = os.environ.copy()
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

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve",
        args.model,
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--pipeline-parallel-size", "1",  # local-only; see module docstring
        "--distributed-executor-backend", "mp",
        "--worker-cls", "vllm.transport.pp_worker.TransportPPWorker",
        "--dtype", args.dtype,
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--host", args.host,
        "--port", str(args.port),
    ]
    if args.quantization:
        cmd += ["--quantization", args.quantization]
    if args.max_model_len:
        cmd += ["--max-model-len", str(args.max_model_len)]
    if args.trust_remote_code:
        cmd += ["--trust-remote-code"]
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

    os.execvpe(cmd[0], cmd, env)
    return 0  # unreachable - execvpe replaces this process on success


if __name__ == "__main__":
    sys.exit(main())
