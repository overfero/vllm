"""Runtime patches for two real, reproducible bugs in
`humming-kernels==0.1.10` that blocked the original GPT-OSS-120B/T4
deployment (README_LIVE_DEPLOYMENT_LOG.md, sibling vllm/ checkout), found
by tracing the exact NVRTC/ptxas compile failures all the way to their
root cause in `humming`'s own (readable, non-obfuscated) Python + CUDA
source, on real Tesla T4 (sm_75) hardware.

This module does NOT edit the installed `humming` package in place
(deliberately - editing files outside this project's own working
directory is a different, larger-blast-radius action than patching code
this project owns, and wasn't done without a human decision to do so).
Both fixes are runtime monkeypatches, applied on `import humming_fix.patch`.

Usage: `import humming_fix.patch` before anything constructs a
`HummingKernel`/loads a Humming-quantized model.


===========================================================================
Fix 1 - num_write_splits vs. block/warp shape mismatch (MoE and dense)
===========================================================================

Root cause: `humming/include/humming/epilogue/pipeline.cuh`,
`EpiloguePipeline::call()`:

    if constexpr (kNumWriteSplits > 1) {
      static_assert(BlockShape::M == WarpShape::M);
      static_assert(BlockShape::M % 32 == 0);
      static_assert(!TuningConfig::kUseTmaC);
    }

`humming/tune/base.py`'s `DeviceHeuristics.get_config()` can, through two
independent downstream branches, produce a final `block_shape_m`/
`warp_shape_m` pair that violates one of the first two asserts above
without ever re-checking whether the `num_write_splits` value it inherited
from `get_base_config()` (e.g. `num_write_splits=2` from several
sm75.py/sm8x.py branches) is still valid for the shapes it ends up with:

  - MoE branch (`meta.num_experts` set - true for GPT-OSS): can select
    `block_shape_m` from `{16, 32, 48, 64}`, two of which (16, 48) are not
    32-aligned -> violates `BlockShape::M % 32 == 0`. This is what the
    original GPT-OSS-120B/T4 deployment hit.

  - Dense branch (`num_warps_m == 2: warp_shape_m = block_shape_m // 2`):
    found in this session via a real DENSE fp16xint4 GEMM correctness run
    on this same shape family, which produced `block_shape_m=64,
    warp_shape_m=32` -> violates `BlockShape::M == WarpShape::M`, a
    *different* assert in the same `if constexpr` block, independent of
    the MoE branch above.

Both are the same underlying problem (the heuristic never re-validates
`num_write_splits` against its own final shape choices), so the fix
checks both invariants together, once, after all of `get_config`'s shape
adjustments are done: when `num_write_splits > 1` but the final shapes
violate either assert, fall back to `num_write_splits = 1` - an
already-supported, simpler epilogue path used by every base config that
never requests splitting at all.

Verified: `humming_fix/test_repro.py`'s
`test_bug1_fixed_num_write_splits_matches_block_shape_m` sweeps
`shape_m` against GPT-OSS's real MoE shape; `correctness_dense.py` (see
`sm75_mma_probe/`) additionally exercises the dense branch end-to-end
through a real NVRTC compile + kernel launch + numerical check.


===========================================================================
Fix 2 - m16n8k16 MMA instruction does not exist on sm_75 (Turing)
===========================================================================

Root cause: `humming/kernel/humming.py`'s
`HummingKernel.select_mma_op_class()` computed, unconditionally,

    mma_shape_k = 256 // self.a_dtype.num_bits

which for a 16-bit `a_dtype` (float16/bfloat16 - GPT-OSS's activation
dtype) gives `mma_shape_k = 16`, selecting the `m16n8k16` MMA instruction
via `humming/config/mma.py`'s `MmaOpClassImpl.generate_ptx()`. Confirmed,
independent of Fix 1, on the exact same GPT-OSS-120B shapes
(shape_n=3072, shape_k=2944, num_experts=128, int4 weight, real bias):

    ptxas ... error: Feature '.m16n8k16' requires .target sm_80 or higher

Confirmed empirically (not from documentation) that this is a real
hardware limit, not a humming-specific restriction: a minimal raw-PTX
probe (`sm75_mma_probe/probe.py`), compiled directly with `nvcc
-arch=sm_75` outside of humming entirely, shows `mma.sync.aligned.
m16n8k16...` is REJECTED by ptxas with this exact error for both f16 and
f32 accumulator, while `mma.sync.aligned.m16n8k8...` (same M=16, N=8,
only K halved from 16 to 8) COMPILES CLEANLY for both. This matches
Turing's real 2nd-generation Tensor Core capabilities: `m16n8k16` is an
Ampere+ (sm_80) instruction; Turing's fp16 mma tops out at `m16n8k8`.

An earlier attempt in this session to fix this (mirroring this file's
existing `sm_version==75 and a_dtype==int8: mma_shape_m=8` special case,
extended to 16-bit `a_dtype`) was CONFIRMED WRONG: it also changes
`mma_shape_m` to 8, producing `m8n8k8`, which is not a valid PTX MMA
shape at all (`ptxas: "Unknown modifier '.m8n8k8'"`). The int8 case's
`mma_shape_m=8` is specific to int8's own valid shape family
(`m8n8k16`/`m8n8k32`), not a generic "sm75 needs mma_shape_m=8" rule -
applying it to fp16 does not transfer. That attempt was removed rather
than shipped.

The correct fix, verified in this session: for sm_75 with a 16-bit
`a_dtype`, leave `mma_shape_m=16` (unchanged) and only set
`mma_shape_k=8`. This requires NO change anywhere in the C++ kernel:
`humming/config/mma.py`'s `MmaOpClassImpl` computes PTX register counts
and the instruction string generically from `(m, n, k, dtypes)`
(`calc_reg_count`, `f"mma.sync.aligned.m{m}n{n}k{k}..."`), and
`humming/include/humming/mma/wmma.cuh`'s `WMMA::run()` already loops
`for k in range(kPartMmaShapeK / MmaShape::K)` - i.e. it already issues
multiple `mma.fma()` calls per warp-K-tile generically; halving
`mma_shape_k` from 16 to 8 simply doubles that loop's trip count to cover
the same K range with two `m16n8k8` instructions instead of one
`m16n8k16` instruction. Confirmed by reading both files directly, then
verified for real:

  - `MmaOpClassImpl(16, 8, 8, float16, float16, float16/float32)`
    produces exactly the PTX confirmed to compile by the raw-PTX probe
    above (same register counts, same instruction text).
  - A real, full `HummingKernel.prepare_kernels(...)` call with GPT-OSS's
    exact MoE shape (shape_n=3072, shape_k=2944, num_experts=128, int4,
    bias=True, grouped_contiguous) - the same call that previously hit
    the `m16n8k16` ptxas error - now compiles AND `cuModuleLoad`s
    successfully on real T4 hardware.
  - A real dense fp16(activations) x int4(weights) GEMM
    (`sm75_mma_probe/correctness_dense.py`, M=128 N=3072 K=2944,
    group_size=32) run end-to-end through `humming.ops.humming_gemm()` on
    real T4 hardware produces output within int4-quantization-noise
    tolerance of a float32 reference matmul - i.e. the corrected MMA
    instruction computes numerically correct results, not just "compiles".

This is now believed to fully resolve both real, independent SM75/T4
blockers found for GPT-OSS-120B in this session. It does NOT prove the
full GPT-OSS-120B model loads and serves correctly end-to-end through
vLLM/SGLang on T4 - that is real, larger, untested scope (real weight
loading, all layer shapes actually present in the model, all GemmTypes
actually exercised, performance). See README.md and
docs/ARCHITECTURE_DECISION.md Part 7 Phase 5/6 for exact scope.
"""
from __future__ import annotations

import functools

from humming import dtypes
from humming.tune.base import DeviceHeuristics
from humming.kernel.humming import HummingKernel
from humming.config import MmaType

_orig_get_config = DeviceHeuristics.get_config.__func__


@functools.wraps(_orig_get_config)
def _patched_get_config(cls, meta, shape_m, use_f16_accum=False, use_batch_invariant=False, gemm_type=None):
    from humming.config import GemmType

    if gemm_type is None:
        gemm_type = GemmType.DENSE

    config = _orig_get_config(
        cls,
        meta=meta,
        shape_m=shape_m,
        use_f16_accum=use_f16_accum,
        use_batch_invariant=use_batch_invariant,
        gemm_type=gemm_type,
    )

    block_shape_m = config["block_shape"][0]
    warp_shape_m = config["warp_shape"][0]
    if config.get("num_write_splits", 1) > 1 and (block_shape_m % 32 != 0 or block_shape_m != warp_shape_m):
        config = dict(config)
        config["num_write_splits"] = 1

    return config


DeviceHeuristics.get_config = classmethod(_patched_get_config)


def _patched_select_mma_op_class(self):
    from humming.config import MmaOpClass

    if self.a_dtype in [dtypes.int4, dtypes.int8]:
        mma_cd_dtype = dtypes.int32
    elif self.use_f16_accum:
        mma_cd_dtype = self.c_dtype
    else:
        mma_cd_dtype = dtypes.float32

    mma_shape_m = self.warp_shape[0] if self.mma_type == MmaType.WGMMA else 16
    mma_shape_n = 64 if self.mma_type == MmaType.WGMMA else 8
    mma_shape_k = 256 // self.a_dtype.num_bits
    if self.sm_version == 75 and self.a_dtype == dtypes.int8:
        mma_shape_m = 8

    # Fix 2 (see module docstring): sm_75 has no m16n8k16 for 16-bit
    # inputs (Ampere+ only) but does have m16n8k8 - keep mma_shape_m=16,
    # only halve mma_shape_k. The generic C++ register-count/PTX-string
    # codegen and the WMMA k-loop both already parameterize on this
    # shape, so no C++ change is needed.
    if self.sm_version == 75 and self.mma_type == MmaType.MMA and self.a_dtype in (dtypes.float16, dtypes.bfloat16):
        mma_shape_k = 8

    if self.mma_type == MmaType.MMA and self.warp_shape[0] % 16 == 8:
        mma_shape_m = 8

    if self.mma_type == MmaType.MMA and mma_shape_m == 8:
        mma_shape_k = mma_shape_k // 2

    if self.mma_type == MmaType.WGMMA:
        assert self.warp_shape[0] % mma_shape_m == 0
        assert self.warp_shape[1] % (mma_shape_n // 4) == 0
    else:
        assert self.warp_shape[0] % mma_shape_m == 0
        assert self.warp_shape[1] % mma_shape_n == 0
    assert self.warp_shape[2] % mma_shape_k == 0

    return MmaOpClass.from_config(
        self.mma_type,
        mma_shape_m,
        mma_shape_n,
        mma_shape_k,
        self.a_dtype,
        self.a_dtype,
        mma_cd_dtype,
    )


HummingKernel.select_mma_op_class = _patched_select_mma_op_class

_PATCH_APPLIED = True
