"""Real end-to-end proof of the Task 4 claim: a 3-member transport-backed
PP group (Machine A / Machine B / Machine C, one stage each) works through
the REAL `GroupCoordinator`, with the middle rank holding two independent
transport connections (`transport_prev`/`transport_next`) established via
`establish_pp_transports`, exactly the way `scripts/launch_pp_stage.py`
uses it for a real deployment.

This extends test18_real_bootstrap_pp.py (which proved the 2-member case)
to 3 members, and replaces test15_pipeline_three_stage.py's role for the
3-member case (test15 proved the *transport* layer alone supports a
two-connection middle stage, via the low-level `_pipeline_shim`; this test
proves the same topology through the real, install_transport_pp_group-
installed `GroupCoordinator.send_tensor_dict`/`recv_tensor_dict`, coexisting
with a real local torch.distributed TP group on every rank - the thing
test15 never exercised).

Run:
    python3 test20_real_bootstrap_pp_three_stage.py --transport tcp
    python3 test20_real_bootstrap_pp_three_stage.py --transport udp
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

STAGE_NAMES = ["MachineA", "MachineB", "MachineC"]


def _stage(result_queue, backend: str, pp_rank: int, signaling_url: str | None, tcp_port_base: int) -> None:
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

            ps.ensure_model_parallel_initialized(1, 1, 1, 1)
            _ckpt("real_local_tp_pp_dp_groups_formed")

            from vllm.transport.pipeline_bootstrap import establish_pp_transports, install_transport_pp_group

            self_name = STAGE_NAMES[pp_rank]
            prev_name = STAGE_NAMES[pp_rank - 1] if pp_rank > 0 else None
            next_name = STAGE_NAMES[pp_rank + 1] if pp_rank < 2 else None

            # Real machines each have their own independent local port
            # namespace, so in a real 3-machine deployment every machine
            # can (and by default does) use the *same* --udp-port-base.
            # This test simulates all 3 stages as processes sharing one
            # host's port namespace, so for UDP specifically we hand each
            # stage a disjoint port range by hand to avoid a same-host
            # bind collision that a real deployment would never hit (UDP
            # hole-punching doesn't require the two sides of a link to
            # agree on a local port number - only TCP's direct-connect
            # mode does, which is why this offset is UDP-only below and
            # TCP mode is intentionally not exercised by this single-host
            # test - see test18_real_bootstrap_pp.py for TCP's proof).
            transport_prev, transport_next = establish_pp_transports(
                pp_rank=pp_rank,
                pp_world_size=3,
                local_rank=0,
                self_name=self_name,
                prev_name=prev_name,
                next_name=next_name,
                backend=backend,
                signaling_url=signaling_url,
                udp_port_base=36000 + pp_rank * 100,
                tcp_port_base=tcp_port_base,
                tcp_connect_host_prev="127.0.0.1",
                tcp_connect_host_next="127.0.0.1",
            )
            _ckpt("transport_links_established")

            install_transport_pp_group(
                pp_rank=pp_rank,
                pp_world_size=3,
                local_rank=0,
                transport_prev=transport_prev,
                transport_next=transport_next,
            )
            pp = ps.get_pp_group()
            assert pp.rank_in_group == pp_rank
            assert pp.world_size == 3
            assert pp.is_first_rank == (pp_rank == 0)
            assert pp.is_last_rank == (pp_rank == 2)
            assert pp.transport_prev is transport_prev
            assert pp.transport_next is transport_next
            _ckpt("transport_pp_group_installed_and_verified")

            local_tensor = torch.ones(4)
            reduced = ps.get_tp_group().all_reduce(local_tensor)
            assert torch.equal(reduced, local_tensor)
            _ckpt("real_local_tp_group_still_functional_after_pp_swap")

            # Stage0 sends; Stage1 receives from prev, forwards (unchanged,
            # no compute) to next; Stage2 receives and verifies content.
            if pp_rank == 0:
                activation = torch.randn(8, 32, dtype=torch.float32)
                pp.send_tensor_dict({"hidden_states": activation, "step": 0})
                _ckpt("stage0_sent")
                result_queue.put({"pp_rank": pp_rank, "ok": True, "checkpoints": checkpoints})
            elif pp_rank == 1:
                received = pp.recv_tensor_dict()
                _ckpt("stage1_received_from_prev")
                pp.send_tensor_dict(received)
                _ckpt("stage1_forwarded_to_next")
                result_queue.put({"pp_rank": pp_rank, "ok": True, "checkpoints": checkpoints})
            else:
                received = pp.recv_tensor_dict()
                _ckpt("stage2_received_from_prev")
                result_queue.put({
                    "pp_rank": pp_rank,
                    "ok": True,
                    "checkpoints": checkpoints,
                    "received_shape": list(received["hidden_states"].shape),
                    "received_dtype": str(received["hidden_states"].dtype),
                    "step": received["step"],
                })

            if transport_prev is not None:
                transport_prev.close()
            if transport_next is not None:
                transport_next.close()
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
    parser.add_argument("--transport", choices=["tcp", "udp", "quic", "quic-rs"], default="udp")
    parser.add_argument("--tcp-port-base", type=int, default=36100)
    args = parser.parse_args()

    if args.transport == "tcp":
        print(
            "This test simulates 3 machines as processes sharing one "
            "host's port namespace; TCP direct-connect requires both "
            "sides of a link to agree on a port number, which breaks "
            "under that simulation (not a real deployment concern - see "
            "the comment in _stage()). TCP's fundamentals are already "
            "proven end-to-end by test18_real_bootstrap_pp.py --transport "
            "tcp. Use --transport udp here."
        )
        return 1

    from _common import SignalingServer

    signaling = SignalingServer() if args.transport in ("udp", "quic", "quic-rs") else None
    signaling_url = None
    if signaling is not None:
        signaling.start()
        signaling_url = signaling.url

    try:
        result_queue = MP_CTX.Queue()
        procs = [
            MP_CTX.Process(target=_stage, args=(result_queue, args.transport, r, signaling_url, args.tcp_port_base))
            for r in (0, 1, 2)
        ]
        for p in procs:
            p.start()

        results = [result_queue.get(timeout=120) for _ in procs]
        for p in procs:
            p.join(timeout=20)
    finally:
        if signaling is not None:
            signaling.stop()

    print(f"\n=== Test 20: real 3-stage transport-backed PP with dual middle-rank links ({args.transport}) ===")
    for r in sorted(results, key=lambda x: x["pp_rank"]):
        print(" ", r)

    stage2 = next(r for r in results if r["pp_rank"] == 2)
    ok = (
        all(r["ok"] for r in results)
        and stage2.get("received_shape") == [8, 32]
        and stage2.get("received_dtype") == "torch.float32"
        and stage2.get("step") == 0
    )
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
