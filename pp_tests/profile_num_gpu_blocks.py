"""One-off: construct a real EngineCore for one PP stage WITHOUT
--num-gpu-blocks-override, to observe the naturally auto-profiled
num_gpu_blocks on real hardware. Used to pick a safe, uniform
--num-gpu-blocks-override value across all 3 machines (Blocker 3,
pp_tests/BLOCKER_REPORT.md) - the override must be <= the smallest
stage's naturally-available block count, since a stage with less free
memory than the override would OOM.

Usage:
    python3 profile_num_gpu_blocks.py --model /data/stage0-checkpoint \
        --pp-rank 0 --pp-world-size 3 --self-name MachineA --next-name MachineB
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

_patch_path = Path(__file__).resolve().parents[1] / "humming_fix" / "patch.py"
_spec = importlib.util.spec_from_file_location("humming_fix_patch", _patch_path)
_patch_module = importlib.util.module_from_spec(_spec)
sys.modules["humming_fix_patch"] = _patch_module
_spec.loader.exec_module(_patch_module)

import torch  # noqa: E402

torch.zeros(1, device="cuda")
torch.cuda.synchronize()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tensor-parallel-size", type=int, default=2)
    p.add_argument("--quantization", default="gptq")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--pp-rank", type=int, default=0)
    p.add_argument("--pp-world-size", type=int, default=3)
    p.add_argument("--language-model-only", action="store_true")
    p.add_argument("--max-model-len", type=int, default=None)
    args = p.parse_args()

    os.environ["PROFILE_PP_RANK"] = str(args.pp_rank)
    os.environ["PROFILE_PP_WORLD_SIZE"] = str(args.pp_world_size)

    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    # Real pp_rank/pp_world_size installed via ProfileOnlyWorker (see
    # _profile_worker.py) - critical: without this, get_pp_group() stays
    # the default world_size=1 group and make_layers() constructs ALL 36
    # layers regardless of pp_rank/world_size or what's actually in the
    # checkpoint, making memory numbers meaningless (this is exactly what
    # happened on the first profiling attempt - looked like a ~14.5GB
    # fixed cost "regardless of layer count" because it was actually
    # always constructing 36 layers both times).
    engine_args = EngineArgs(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=1,
        distributed_executor_backend="mp",
        worker_cls="_profile_worker.ProfileOnlyWorker",
        dtype=args.dtype,
        quantization=args.quantization,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        async_scheduling=False,
        language_model_only=args.language_model_only,
        max_model_len=args.max_model_len,
    )
    vllm_config = engine_args.create_engine_config()
    engine_core = EngineCore(vllm_config=vllm_config, executor_class=MultiprocExecutor, log_stats=False)

    kv_cache_configs = engine_core.scheduler.kv_cache_config if hasattr(engine_core.scheduler, "kv_cache_config") else None
    num_blocks = None
    for attr in ("num_gpu_blocks", "num_blocks"):
        if hasattr(engine_core.scheduler, attr):
            num_blocks = getattr(engine_core.scheduler, attr)
            break
    print(f"PROFILE_RESULT: num_gpu_blocks={num_blocks} kv_cache_config={kv_cache_configs}")
    engine_core.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
