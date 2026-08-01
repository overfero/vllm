"""Real end-to-end test: two GPU processes, each simulating one machine,
each with a REAL local torch.distributed bootstrap (loopback, real NCCL-
capable single-GPU group - standard, unmodified vLLM `Worker.init_device()`),
with the module-global `_PP` pipeline-parallel group then replaced by a
transport-backed synthetic group (`vllm/transport/pipeline_bootstrap.py`)
before the model is loaded. `make_layers()`/`is_first_rank`/`is_last_rank`
all read `get_pp_group()`, so the real openai-community/gpt2 model should
split itself across the two stages according to *our* synthetic group,
with zero model-code changes.

This is a staged test - each stage prints a clear checkpoint, so a failure
partway through still reports exactly how far real vLLM code got.

Run:
    python3 test17_real_gpu_pipeline.py --transport tcp
    python3 test17_real_gpu_pipeline.py --transport udp
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # make `vllm` importable
sys.path.insert(0, str(Path(__file__).resolve().parent))  # make `_env_stubs`/`_common` importable

MODEL_PATH = os.environ.get("VLLM_TEST_GPT2_PATH", "/models/gpt2")
MP_CTX = mp.get_context("spawn")  # CUDA-safe, unlike fork


def _stage(result_queue, backend: str, cuda_index: int, pp_rank: int, signaling_url: str | None) -> None:
    import _env_stubs  # noqa: F401  (must be first - see its docstring)

    checkpoints: list[str] = []

    def _ckpt(name: str) -> None:
        checkpoints.append(name)
        print(f"[stage{pp_rank}] checkpoint: {name}", flush=True)

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_index)

        import torch

        from vllm.config import VllmConfig
        from vllm.engine.arg_utils import EngineArgs

        engine_args = EngineArgs(
            model=MODEL_PATH,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,  # trivial LOCAL pp - see pipeline_bootstrap.py docstring
            enforce_eager=True,  # avoid CUDA graph capture on our synthetic PP group (documented limitation)
            gpu_memory_utilization=0.5,
            max_model_len=64,
            dtype="float16",
        )
        vllm_config: VllmConfig = engine_args.create_engine_config()
        _ckpt("vllm_config_built")

        from vllm.config import set_current_vllm_config
        from vllm.utils.network_utils import get_distributed_init_method, get_loopback_ip, get_open_port
        from vllm.v1.worker.gpu_worker import Worker

        with set_current_vllm_config(vllm_config):
            distributed_init_method = get_distributed_init_method(get_loopback_ip(), get_open_port())
            worker = Worker(
                vllm_config=vllm_config,
                local_rank=0,
                rank=0,
                distributed_init_method=distributed_init_method,
                is_driver_worker=True,
            )
            worker.init_device()  # real local torch.distributed bootstrap (loopback) - unmodified vLLM code
            _ckpt("real_local_torch_distributed_bootstrap_ok")

            import vllm.distributed.parallel_state as ps

            assert torch.distributed.is_initialized()
            assert ps.get_tp_group().world_size == 1
            _ckpt("real_tp_group_confirmed")

            # --- swap in the transport-backed cross-machine PP group ---
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from _common import SignalingServer, transport_config_pair  # noqa: E402

            from vllm.transport import get_transport
            from vllm.transport.pipeline_bootstrap import install_transport_pp_group

            cfg0, cfg1 = transport_config_pair(backend, "GpuStage0", "GpuStage1", signaling_url, 34000)
            my_cfg = cfg0 if pp_rank == 0 else cfg1
            transport = get_transport(backend)
            transport.connect(my_cfg)
            _ckpt("transport_connected")

            install_transport_pp_group(transport, pp_rank=pp_rank, pp_world_size=2, local_rank=0)
            assert ps.get_pp_group().transport is transport
            assert ps.get_pp_group().world_size == 2
            _ckpt("transport_pp_group_installed")

            worker.load_model()
            _ckpt("model_loaded")

            model = worker.model_runner.model
            pp = ps.get_pp_group()
            num_layers = len(model.transformer.h) if hasattr(model, "transformer") else None
            result_queue.put({
                "pp_rank": pp_rank,
                "ok": True,
                "checkpoints": checkpoints,
                "is_first_rank": pp.is_first_rank,
                "is_last_rank": pp.is_last_rank,
                "start_layer": model.transformer.start_layer,
                "end_layer": model.transformer.end_layer,
                "num_instantiated_layers": sum(
                    1 for m in model.transformer.h if type(m).__name__ != "PPMissingLayer"
                ),
            })

            # --- real CUDA-tensor activation transfer across the transport ---
            if pp_rank == 0:
                real_tensor = torch.randn(4, 16, dtype=torch.float16, device="cuda")
                pp.send_tensor_dict({"hidden_states": real_tensor, "note": "from stage0"})
                _ckpt("sent_real_cuda_tensor_via_transport")
            else:
                received = pp.recv_tensor_dict()
                _ckpt("received_real_cuda_tensor_via_transport")
                result_queue.put({
                    "pp_rank": pp_rank,
                    "role": "tensor_transfer_check",
                    "received_device": str(received["hidden_states"].device),
                    "received_shape": list(received["hidden_states"].shape),
                    "note": received["note"],
                })

            worker.check_health() if hasattr(worker, "check_health") else None
            transport.close()

    except Exception as exc:  # noqa: BLE001 - report exactly how far we got
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

    from _common import SignalingServer, free_port  # noqa: E402

    signaling = SignalingServer() if args.transport == "udp" else None
    signaling_url = None
    if signaling is not None:
        signaling.start()
        signaling_url = signaling.url

    try:
        result_queue = MP_CTX.Queue()
        p0 = MP_CTX.Process(target=_stage, args=(result_queue, args.transport, 0, 0, signaling_url))
        p1 = MP_CTX.Process(target=_stage, args=(result_queue, args.transport, 1, 1, signaling_url))
        p0.start()
        p1.start()

        results = []
        for _ in range(3):  # each stage puts >=1 result; stage1 puts 2 (load + tensor check)
            try:
                results.append(result_queue.get(timeout=180))
            except Exception:
                break
        p0.join(timeout=30)
        p1.join(timeout=30)
    finally:
        if signaling is not None:
            signaling.stop()

    print(f"\n=== Test 17: real 2-GPU pipeline bootstrap + model load ({args.transport}) ===")
    ok = True
    for r in results:
        print(f"  {r}")
        if r.get("ok") is False:
            ok = False
            print(f"  --- traceback (pp_rank={r['pp_rank']}) ---")
            print(r.get("traceback", ""))

    print("PASS" if ok else "FAIL (see checkpoints above for exactly how far real vLLM code got)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
