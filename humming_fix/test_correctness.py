"""Real, on-hardware numerical correctness check for Fix 2 (see patch.py):
does the corrected m16n8k8 (mma_shape_m=16, mma_shape_k=8) MMA instruction
actually compute the right answer, not just compile? Runs a real dense
fp16(activations) x int4(weights) GEMM end-to-end through
`humming.ops.humming_gemm()` on real T4 hardware and compares against a
float32 reference matmul. `select_mma_op_class()` doesn't branch on
MoE-vs-dense (only on sm_version/a_dtype), so this dense case exercises
the identical MMA instruction path GPT-OSS's MoE layers use, with a much
simpler calling convention (no expert routing) than the MoE case in
`test_repro.py`.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA device")


@pytest.mark.skipif(
    torch.cuda.get_device_capability(0) != (7, 5) if torch.cuda.is_available() else True,
    reason="Fix 2 is SM75(T4)-specific; on other SM versions this path isn't exercised",
)
def test_bug2_fix_produces_correct_dense_gemm_output():
    import humming_fix.patch  # noqa: F401  (Fix 1 + Fix 2, both applied)

    from humming import dtypes, ops
    from humming.config import MmaType
    from humming.layer import HummingLayerMeta
    from humming.ops.weight import pack_weight
    from humming.utils.test import generate_random_bias, generate_random_inputs, generate_random_weight
    from humming.utils.weight import prepare_humming_bias, prepare_humming_weight, prepare_humming_weight_scale

    torch.cuda.init()
    torch.zeros(1, device="cuda")
    torch.manual_seed(0)

    m, n, k = 128, 3072, 2944
    group_size = 32

    _, inputs_ref, inputs, _ = generate_random_inputs(m, k, dtype=dtypes.float16)
    _, weight_ref, quanted_weight, weight_scale, _, _ = generate_random_weight(
        n, k, group_size, dataclasses.replace(dtypes.int4, is_signed=False),
        scale_dtype=dtypes.float16, has_zero_point=False,
    )
    bias = generate_random_bias(n, dtypes.float16)

    meta = HummingLayerMeta(
        shape_n=n,
        shape_k=k,
        num_experts=0,
        b_dtype=dataclasses.replace(dtypes.int4, is_signed=False),
        a_dtype=dtypes.float16,
        c_dtype=dtypes.float16,
        bs_dtype=dtypes.float16,
        weight_scale_group_size=group_size,
        has_bias=True,
    )

    weight_packed_int = pack_weight(quanted_weight.to(torch.int32).contiguous(), num_bits=4)
    weight_packed = prepare_humming_weight(
        weight=weight_packed_int,
        b_dtype=meta.b_dtype,
        a_dtype=meta.a_dtype,
        zero_point=None,
        use_wgmma=meta.mma_type == MmaType.WGMMA,
        use_fused_e8m0_scale=meta.use_fused_e8m0_scale,
        packed=True,
    )
    weight_scale_packed = prepare_humming_weight_scale(
        weight_scale, to_apply_on_c=meta.should_apply_bs_on_c, is_blockwise=False,
    )
    bias_packed = prepare_humming_bias(bias)
    locks = torch.zeros((1024,), dtype=torch.int32, device="cuda")

    out = ops.humming_gemm(
        meta.to_str(), None, None,
        inputs, weight_packed,
        weight_scale=weight_scale_packed,
        bias=bias_packed,
        locks=locks,
    )

    ref = inputs_ref @ weight_ref.transpose(-1, -2) + bias.float()
    out_f = out.float()

    # Not bit-exact - fp16 + int4 quantization noise is expected. This
    # checks the MMA math is doing real matrix multiplication (a broken
    # MMA op class / register layout produces wildly wrong output, far
    # outside this tolerance, not merely quantization-level noise).
    assert torch.allclose(out_f, ref, atol=0.5, rtol=0.1), (
        f"max abs err {(out_f - ref).abs().max().item():.4f} exceeds "
        "int4-quantization-noise tolerance"
    )
