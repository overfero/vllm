"""Real 3-SEPARATE-MACHINE version of test20_real_bootstrap_pp_three_stage.py
- not a local-process simulation. Each of Machine A/B/C runs this script
independently with its own --pp-rank, using a real public signaling
server (zrok-tunneled) reachable over each machine's own real internet
path, exercising genuine UDP hole punching between 3 distinct hosts with
3 distinct public IPs (confirmed via curl api.ipify.org: A, B, C all
different).

Run identically to test20's per-stage logic, but as one process per real
machine instead of one process per pp_rank on a single host.
"""
from __future__ import annotations

import argparse
import json
import sys
import types


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pp-rank", type=int, required=True)
    p.add_argument("--signaling-url", required=True)
    args = p.parse_args()

    STAGE_NAMES = ["MachineA9838", "MachineB9838", "MachineC9838"]
    pp_rank = args.pp_rank
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

            transport_prev, transport_next = establish_pp_transports(
                pp_rank=pp_rank,
                pp_world_size=3,
                local_rank=0,
                self_name=self_name,
                prev_name=prev_name,
                next_name=next_name,
                backend="udp",
                signaling_url=args.signaling_url,
                udp_port_base=36000,
                connect_timeout=90.0,
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
            _ckpt("transport_pp_group_installed_and_verified")

            local_tensor = torch.ones(4)
            reduced = ps.get_tp_group().all_reduce(local_tensor)
            assert torch.equal(reduced, local_tensor)
            _ckpt("real_local_tp_group_still_functional_after_pp_swap")

            if pp_rank == 0:
                activation = torch.randn(8, 32, dtype=torch.float32)
                pp.send_tensor_dict({"hidden_states": activation, "step": 0})
                _ckpt("stage0_sent")
                result = {"pp_rank": pp_rank, "ok": True, "checkpoints": checkpoints}
            elif pp_rank == 1:
                received = pp.recv_tensor_dict()
                _ckpt("stage1_received_from_prev")
                pp.send_tensor_dict(received)
                _ckpt("stage1_forwarded_to_next")
                result = {"pp_rank": pp_rank, "ok": True, "checkpoints": checkpoints}
            else:
                received = pp.recv_tensor_dict()
                _ckpt("stage2_received_from_prev")
                result = {
                    "pp_rank": pp_rank,
                    "ok": True,
                    "checkpoints": checkpoints,
                    "received_shape": list(received["hidden_states"].shape),
                    "received_dtype": str(received["hidden_states"].dtype),
                    "step": received["step"],
                }

            if transport_prev is not None:
                transport_prev.close()
            if transport_next is not None:
                transport_next.close()

    except Exception as exc:  # noqa: BLE001
        import traceback

        result = {
            "pp_rank": pp_rank,
            "ok": False,
            "checkpoints": checkpoints,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    print("RESULT_JSON: " + json.dumps(result), flush=True)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
