"""Real reproduction/verification tests for both humming-kernels T4/SM75
bugs documented in README.md - both now fixed (see patch.py). Requires
real CUDA hardware (T4/SM75 specifically for the Bug 2 test) and
`humming-kernels==0.1.10` installed.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA device")

# GPT-OSS-120B's real MoE GEMM shape (see README_LIVE_DEPLOYMENT_LOG.md,
# sibling vllm/ checkout, for how these were derived from the real
# checkpoint's config.json / the original NVRTC error's ProblemShape).
_GPT_OSS_SHAPE_N = 3072
_GPT_OSS_SHAPE_K = 2944
_GPT_OSS_NUM_EXPERTS = 128


def _build_meta():
    from humming import dtypes
    from humming.layer import HummingLayerMeta

    return HummingLayerMeta(
        shape_n=_GPT_OSS_SHAPE_N,
        shape_k=_GPT_OSS_SHAPE_K,
        num_experts=_GPT_OSS_NUM_EXPERTS,
        b_dtype=dataclasses.replace(dtypes.int4, is_signed=False),
        a_dtype=dtypes.float16,
        c_dtype=dtypes.float16,
        weight_scale_group_size=32,
        has_bias=True,
    )


@pytest.mark.parametrize("shape_m", [1, 8, 48, 96, 128, 256])
def test_bug1_fixed_num_write_splits_matches_block_shape_m(shape_m: int) -> None:
    """Bug 1 (see README.md): before the fix, some shape_m values produce
    num_write_splits=2 with a non-32-aligned block_shape_m - the exact
    combination humming's own epilogue pipeline template rejects at NVRTC
    compile time. After humming_fix.patch is applied, this combination
    must never occur."""
    import humming_fix.patch  # noqa: F401  (applies Bug 1's fix)
    from humming.config import GemmType
    from humming.tune.sm75 import Sm75Heuristics

    meta = _build_meta()
    config = Sm75Heuristics.get_config(meta=meta, shape_m=shape_m, gemm_type=GemmType.GROUPED_CONTIGUOUS)
    block_shape_m = config["block_shape"][0]
    num_write_splits = config.get("num_write_splits", 1)

    assert not (num_write_splits > 1 and block_shape_m % 32 != 0), (
        f"shape_m={shape_m}: block_shape_m={block_shape_m} is not 32-aligned "
        f"but num_write_splits={num_write_splits} > 1 - this combination "
        "violates EpiloguePipeline::call()'s static_assert and will fail "
        "NVRTC compilation. Bug 1's fix should prevent this."
    )


@pytest.mark.skipif(
    torch.cuda.get_device_capability(0) != (7, 5) if torch.cuda.is_available() else True,
    reason="Bug 2's exact error text/fix is SM75(T4)-specific",
)
def test_bug2_fixed_gpt_oss_moe_shape_compiles_and_loads() -> None:
    """Real end-to-end NVRTC compile + cuModuleLoad (not just the
    heuristic) with GPT-OSS's real MoE shape - the exact call that used to
    raise NVRTC_ERROR_COMPILATION on real T4 hardware (m16n8k16 is
    Ampere+-only) now succeeds with Fix 2 applied (patch.py:
    _patched_select_mma_op_class, mma_shape_k=8 for sm75+16-bit a_dtype).
    See test_correctness.py for a real numerical-correctness check of the
    same fix on a simpler dense shape."""
    import humming_fix.patch  # noqa: F401  (Fix 1 + Fix 2)
    from humming import dtypes
    from humming.kernel.humming import HummingKernel

    torch.cuda.init()
    torch.zeros(1, device="cuda")

    layer_config = dict(
        shape_n=_GPT_OSS_SHAPE_N,
        shape_k=_GPT_OSS_SHAPE_K,
        num_experts=_GPT_OSS_NUM_EXPERTS,
        b_dtype=dataclasses.replace(dtypes.int4, is_signed=False),
        a_dtype=dtypes.float16,
        c_dtype=dtypes.float16,
        weight_scale_group_size=32,
        has_bias=True,
    )
    compute_config = dict(gemm_type="grouped_contiguous")

    kernels = HummingKernel.prepare_kernels(layer_config, compute_config, None)
    assert kernels is not None
