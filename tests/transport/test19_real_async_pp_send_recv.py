"""Proves the fix made this session to `vllm/distributed/parallel_state.py`:
real vLLM's actual production PP call site
(`vllm/v1/worker/gpu_worker.py`, `Worker._execute_model` -
`get_pp_group().irecv_tensor_dict(...)` / `.isend_tensor_dict(...)`) uses the
ASYNC isend/irecv primitives, not the synchronous send_tensor_dict/
recv_tensor_dict that test18 exercised. Before this session's fix, only the
synchronous pair checked `self.transport is not None` - the async pair
(what real vLLM actually calls) fell straight through to the real
torch.distributed code path, which would raise on a transport-backed
synthetic PP group (no real `device_group`/`cpu_group` attributes exist on
it - see `pipeline_bootstrap.py`).

Simulates one PP link (2 "machines") each with a real local TP=2 group -
matching this project's actual TP=2/machine topology, not test18's trivial
TP=1 - so 4 processes total: (pp_rank, local_tp_rank) in {0,1}x{0,1}. Each
process independently installs its own transport-backed PP group and
independently calls isend_tensor_dict/irecv_tensor_dict with the exact
argument shape gpu_worker.py uses (`all_gather_group=get_tp_group()`,
`all_gather_tensors={...}`, waiting on returned handles, running returned
postprocess callables) - proving the fix handles the real call convention,
not a simplified direct call, and that each local TP rank gets its own
correct transport link to its counterpart rank on the other "machine".

Run:
    python3 test19_real_async_pp_send_recv.py --transport tcp
    python3 test19_real_async_pp_send_recv.py --transport udp
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

MP_CTX = mp.get_context("spawn")


def _worker(result_queue, backend: str, pp_rank: int, local_rank: int, signaling_url: str | None) -> None:
    import _env_stubs  # noqa: F401  (must be first)

    try:
        import torch

        from vllm.config import set_current_vllm_config
        from vllm.config.parallel import ParallelConfig
        from vllm.utils.network_utils import get_distributed_init_method, get_loopback_ip, get_open_port

        parallel_config = ParallelConfig(tensor_parallel_size=2, pipeline_parallel_size=1, data_parallel_size=1)
        fake_vllm_config = types.SimpleNamespace(
            parallel_config=parallel_config,
            model_config=types.SimpleNamespace(is_moe=False),
            compilation_config=types.SimpleNamespace(mode=None, custom_op_log_check=lambda: None),
        )

        with set_current_vllm_config(fake_vllm_config):
            import vllm.distributed.parallel_state as ps

            # Real local TP=2 rendezvous - independent loopback port per
            # pp_rank (each "machine" has its own local TP group).
            dim = get_distributed_init_method(get_loopback_ip(), 29500 + pp_rank)
            ps.init_distributed_environment(
                world_size=2, rank=local_rank, distributed_init_method=dim,
                local_rank=local_rank, backend="gloo",
            )
            ps.ensure_model_parallel_initialized(2, 1, 1, 1)
            assert ps.get_tp_group().world_size == 2

            from _common import transport_config_pair
            from vllm.transport import get_transport
            from vllm.transport.pipeline_bootstrap import install_transport_pp_group

            # One independent transport link per (pp_rank pair, local_rank) -
            # this is the real topology: each local TP rank on machine A
            # talks to its correspondingly-ranked TP worker on machine B.
            cfg0, cfg1 = transport_config_pair(
                backend, f"T19-A-r{local_rank}", f"T19-B-r{local_rank}",
                signaling_url, 37000 + local_rank * 1000,
            )
            my_cfg = cfg0 if pp_rank == 0 else cfg1
            transport = get_transport(backend)
            transport.connect(my_cfg)

            install_transport_pp_group(transport, pp_rank=pp_rank, pp_world_size=2, local_rank=local_rank)
            pp = ps.get_pp_group()
            tp = ps.get_tp_group()
            assert pp.transport is transport

            if pp_rank == 0:
                activation = torch.arange(16, dtype=torch.float32).reshape(4, 4) + local_rank
                handles = pp.isend_tensor_dict(
                    {"hidden_states": activation, "step": 0, "local_rank": local_rank},
                    all_gather_group=tp,
                    all_gather_tensors={},
                )
                for h in handles:
                    h.wait()
                result_queue.put({
                    "pp_rank": pp_rank, "local_rank": local_rank, "ok": True,
                    "sent_shape": list(activation.shape),
                })
            else:
                tensor_dict, handles, postprocess = pp.irecv_tensor_dict(
                    all_gather_group=tp,
                    all_gather_tensors={},
                )
                for h in handles:
                    h.wait()
                for fn in postprocess:
                    fn()
                result_queue.put({
                    "pp_rank": pp_rank, "local_rank": local_rank, "ok": True,
                    "received_shape": list(tensor_dict["hidden_states"].shape),
                    "received_dtype": str(tensor_dict["hidden_states"].dtype),
                    "step": tensor_dict["step"],
                    "sender_local_rank": tensor_dict["local_rank"],
                })

            transport.close()

    except Exception as exc:  # noqa: BLE001
        import traceback

        result_queue.put({
            "pp_rank": pp_rank, "local_rank": local_rank, "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["tcp", "udp", "quic", "quic-rs"], default="tcp")
    args = parser.parse_args()

    from _common import SignalingServer

    signaling = SignalingServer() if args.transport in ("udp", "quic", "quic-rs") else None
    signaling_url = None
    if signaling is not None:
        signaling.start()
        signaling_url = signaling.url

    try:
        result_queue = MP_CTX.Queue()
        procs = []
        for pp_rank in (0, 1):
            for local_rank in (0, 1):
                p = MP_CTX.Process(
                    target=_worker,
                    args=(result_queue, args.transport, pp_rank, local_rank, signaling_url),
                )
                p.start()
                procs.append(p)

        results = [result_queue.get(timeout=120) for _ in range(4)]
        for p in procs:
            p.join(timeout=30)
    finally:
        if signaling is not None:
            signaling.stop()

    print(f"\n=== Test 19: real async isend_tensor_dict/irecv_tensor_dict via transport ({args.transport}) ===")
    ok = True
    for r in sorted(results, key=lambda x: (x["pp_rank"], x["local_rank"])):
        print(f"  {r}")
        if r.get("ok") is False:
            ok = False
            print(f"  --- traceback (pp_rank={r['pp_rank']}, local_rank={r['local_rank']}) ---\n{r.get('traceback', '')}")

    for r in results:
        if r["pp_rank"] == 1 and r.get("ok"):
            if not (
                r["received_shape"] == [4, 4]
                and r["received_dtype"] == "torch.float32"
                and r["sender_local_rank"] == r["local_rank"]
            ):
                ok = False

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
