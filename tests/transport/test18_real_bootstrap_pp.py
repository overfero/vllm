"""Real end-to-end proof of the core Phase 5 claim: on each of two
processes (simulating two machines), a REAL local torch.distributed
process group is formed via the REAL, unmodified
`init_distributed_environment`/`ensure_model_parallel_initialized`
(loopback rendezvous - always reachable, same machine), and the resulting
trivial local `_PP` is then replaced with a transport-backed synthetic
2-member group (`vllm/transport/pipeline_bootstrap.py`) that crosses to
the *other* process via UDP hole punch (or TCP) - not via torch.distributed.

This avoids vLLM's model-registry subprocess (broken in this sandbox by a
pervasive, pre-existing torch-version incompatibility unrelated to this
project - see README_GPTOSS_120B_UDP.md) by using a minimal, hand-built
config instead of `EngineArgs.create_engine_config()`. Nothing here is
GPT-OSS/model-specific: this is pure distributed-layer proof.

Run:
    python3 test18_real_bootstrap_pp.py --transport tcp
    python3 test18_real_bootstrap_pp.py --transport udp
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


def _stage(result_queue, backend: str, pp_rank: int, signaling_url: str | None) -> None:
    import _env_stubs  # noqa: F401  (must be first)

    checkpoints: list[str] = []

    def _ckpt(name: str) -> None:
        checkpoints.append(name)
        print(f"[stage{pp_rank}] checkpoint: {name}", flush=True)

    try:
        import torch

        from vllm.config import set_current_vllm_config
        from vllm.config.parallel import ParallelConfig
        from vllm.utils.network_utils import get_distributed_init_method, get_loopback_ip, get_open_port

        parallel_config = ParallelConfig(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1)
        fake_vllm_config = types.SimpleNamespace(
            parallel_config=parallel_config,
            model_config=types.SimpleNamespace(is_moe=False),
            compilation_config=types.SimpleNamespace(mode=None, custom_op_log_check=lambda: None),
        )

        with set_current_vllm_config(fake_vllm_config):
            import vllm.distributed.parallel_state as ps

            dim = get_distributed_init_method(get_loopback_ip(), get_open_port())
            ps.init_distributed_environment(
                world_size=1, rank=0, distributed_init_method=dim, local_rank=0, backend="gloo"
            )
            _ckpt("real_local_torch_distributed_init_ok")
            assert torch.distributed.is_initialized()

            ps.ensure_model_parallel_initialized(1, 1, 1, 1)
            _ckpt("real_local_tp_pp_dp_groups_formed")
            assert ps.get_tp_group().world_size == 1
            trivial_pp_rank = ps.get_pp_group().rank
            _ckpt(f"trivial_local_pp_confirmed(rank={trivial_pp_rank})")

            from _common import free_port, transport_config_pair
            from vllm.transport import get_transport
            from vllm.transport.pipeline_bootstrap import install_transport_pp_group

            cfg0, cfg1 = transport_config_pair(backend, "BootstrapA", "BootstrapB", signaling_url, 35000)
            my_cfg = cfg0 if pp_rank == 0 else cfg1
            transport = get_transport(backend)
            transport.connect(my_cfg)
            _ckpt("transport_connected")

            install_transport_pp_group(transport, pp_rank=pp_rank, pp_world_size=2, local_rank=0)
            pp = ps.get_pp_group()
            assert pp.transport is transport
            assert pp.world_size == 2
            assert pp.rank_in_group == pp_rank
            assert pp.is_first_rank == (pp_rank == 0)
            assert pp.is_last_rank == (pp_rank == 1)
            _ckpt("transport_pp_group_installed_and_verified")

            # Real torch.distributed local TP group STILL alive and usable
            # after the PP swap - proves the two coexist without interference.
            local_tensor = torch.ones(4)
            reduced = ps.get_tp_group().all_reduce(local_tensor)
            assert torch.equal(reduced, local_tensor)  # world_size=1: all_reduce is a no-op passthrough
            _ckpt("real_local_tp_group_still_functional_after_pp_swap")

            # The actual cross-machine activation transfer, through the REAL
            # (now-installed) GroupCoordinator.send_tensor_dict/.recv_tensor_dict.
            if pp_rank == 0:
                activation = torch.randn(8, 32, dtype=torch.float32)
                pp.send_tensor_dict({"hidden_states": activation, "step": 0})
                _ckpt("sent_activation_via_real_groupcoordinator_send_tensor_dict")
                result_queue.put({"pp_rank": pp_rank, "ok": True, "checkpoints": checkpoints})
            else:
                received = pp.recv_tensor_dict()
                _ckpt("received_activation_via_real_groupcoordinator_recv_tensor_dict")
                result_queue.put({
                    "pp_rank": pp_rank,
                    "ok": True,
                    "checkpoints": checkpoints,
                    "received_shape": list(received["hidden_states"].shape),
                    "received_dtype": str(received["hidden_states"].dtype),
                    "step": received["step"],
                })

            transport.close()
            ps.destroy_distributed_environment() if hasattr(ps, "destroy_distributed_environment") else None

    except Exception as exc:  # noqa: BLE001
        import traceback

        result_queue.put({
            "pp_rank": pp_rank,
            "ok": False,
            "checkpoints": checkpoints,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["tcp", "udp"], default="tcp")
    args = parser.parse_args()

    from _common import SignalingServer

    signaling = SignalingServer() if args.transport == "udp" else None
    signaling_url = None
    if signaling is not None:
        signaling.start()
        signaling_url = signaling.url

    try:
        result_queue = MP_CTX.Queue()
        p0 = MP_CTX.Process(target=_stage, args=(result_queue, args.transport, 0, signaling_url))
        p1 = MP_CTX.Process(target=_stage, args=(result_queue, args.transport, 1, signaling_url))
        p0.start()
        p1.start()

        results = [result_queue.get(timeout=120), result_queue.get(timeout=120)]
        p0.join(timeout=20)
        p1.join(timeout=20)
    finally:
        if signaling is not None:
            signaling.stop()

    print(f"\n=== Test 18: real distributed bootstrap + transport-backed PP ({args.transport}) ===")
    ok = True
    for r in sorted(results, key=lambda x: x["pp_rank"]):
        print(f"  {r}")
        if r.get("ok") is False:
            ok = False
            print(f"  --- traceback (pp_rank={r['pp_rank']}) ---\n{r.get('traceback', '')}")

    stage1 = next((r for r in results if r["pp_rank"] == 1), None)
    if stage1 and stage1.get("ok"):
        ok = ok and stage1["received_shape"] == [8, 32] and stage1["received_dtype"] == "torch.float32"

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
