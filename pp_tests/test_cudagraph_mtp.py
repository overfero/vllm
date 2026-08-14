"""One-off: does `--enable-cudagraph` + MTP speculative decoding actually
crash the MTP drafter's torch.compile/Dynamo trace, or was that fixed by
something else since it was last checked (the pp_worker.py MTP-patch fix,
the real PYTHONPATH/spawn bug)? Constructs a real EngineCore for one PP
stage exactly like profile_num_gpu_blocks.py (fake PP group via
ProfileOnlyWorker, no real network/peers needed - reaches profile_run()
in seconds instead of the minutes a full cluster launch takes), but with
`--speculative-config` and WITHOUT `--enforce-eager`, isolating just the
question this experiment is about.

Usage:
    python3 pp_tests/test_cudagraph_mtp.py --model /data/stage0-checkpoint \
        --pp-rank 0 --pp-world-size 4
"""
import argparse
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

_patch_path = Path(__file__).resolve().parents[1] / "humming_fix" / "patch.py"
_spec = importlib.util.spec_from_file_location("humming_fix_patch", _patch_path)
_patch_module = importlib.util.module_from_spec(_spec)
sys.modules["humming_fix_patch"] = _patch_module
_spec.loader.exec_module(_patch_module)

# Same real fix as vllm/transport/pp_worker.py - see that module's
# comment for why this can't be done via PYTHONPATH/sitecustomize.
import vllm.transport.qwen35_mtp_pp_fix  # noqa: E402,F401

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
    p.add_argument("--pp-world-size", type=int, default=4)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--num-gpu-blocks-override", type=int, default=60)
    p.add_argument("--max-num-seqs", type=int, default=8)
    args = p.parse_args()

    os.environ["PROFILE_PP_RANK"] = str(args.pp_rank)
    os.environ["PROFILE_PP_WORLD_SIZE"] = str(args.pp_world_size)

    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    engine_args = EngineArgs(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=1,
        distributed_executor_backend="mp",
        worker_cls="_profile_worker.ProfileOnlyWorker",
        dtype=args.dtype,
        quantization=args.quantization,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,  # the whole point: exercise torch.compile/cudagraph
        async_scheduling=False,
        language_model_only=True,
        max_model_len=args.max_model_len,
        num_gpu_blocks_override=args.num_gpu_blocks_override,
        max_num_seqs=args.max_num_seqs,
        speculative_config={"method": "mtp", "num_speculative_tokens": 1},
    )
    vllm_config = engine_args.create_engine_config()
    try:
        engine_core = EngineCore(vllm_config=vllm_config, executor_class=MultiprocExecutor, log_stats=False)
    except Exception as exc:  # noqa: BLE001
        print("RESULT: CRASHED")
        print(f"exception type: {type(exc).__name__}")
        traceback.print_exc()
        return 1

    print("RESULT: SUCCESS - EngineCore constructed with cudagraph + MTP, no crash")
    engine_core.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
