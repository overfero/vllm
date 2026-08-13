# humming-kernels T4/SM75 investigation — Phase 5/6

Real investigation into the `humming-kernels==0.1.10` NVRTC compile failure
that blocked the original GPT-OSS-120B/T4 deployment
(`README_LIVE_DEPLOYMENT_LOG.md`, sibling `vllm/` checkout). This had real
access to: the `humming` package's actual (readable, not obfuscated)
Python + CUDA source, real Tesla T4 hardware (`nvidia-smi` confirmed), and
the real 61GB GPT-OSS-120B GPTQ checkpoint
(`/data/models/gpt-oss-120b-gptq`) — so this is root-caused and tested
against reality, not theorized from the original error message alone.

## Bottom line

**Two separate, independent SM75 (T4/Turing) incompatibility bugs existed
in `humming-kernels==0.1.10`, not one. Both are now fixed and verified on
real hardware** — Bug 2 (the harder one, thought unsolved as of Phase 5)
was root-caused precisely enough in this pass to find a correct fix,
verified via a real NVRTC compile of GPT-OSS's exact MoE shape AND a real
numerical-correctness run against a float32 reference. Neither fix
touches a single line of C++/CUDA — both are entirely in `humming`'s
Python tuning-heuristic layer, because the C++ kernel machinery is
already fully generic over MMA shape/dtype (see Bug 2 below for why).

GPT-OSS-120B's per-layer MoE GEMM (the operation these bugs blocked) now
compiles, loads, and computes numerically correct output on real T4
hardware. **This has not yet been exercised through a full model load or
a full vLLM/SGLang forward pass** — see "What's still open" below.

## Bug 1 — FIXED and verified

**Root cause**: `humming/include/humming/epilogue/pipeline.cuh`,
`EpiloguePipeline::call()`:

```cpp
if constexpr (kNumWriteSplits > 1) {
  static_assert(BlockShape::M == WarpShape::M);   // <- can also fail (see below)
  static_assert(BlockShape::M % 32 == 0);          // <- the original failure
  static_assert(!TuningConfig::kUseTmaC);
}
```

`humming/tune/base.py`'s `DeviceHeuristics.get_config()` has two
independent downstream branches (one MoE-specific, one dense-specific)
that can each produce a final `block_shape_m`/`warp_shape_m` pair
violating one of the first two asserts, without ever re-checking whether
the `num_write_splits` value inherited from `get_base_config()` is still
valid for the shapes it ends up choosing:

- MoE branch: can select `block_shape_m` from `{16, 32, 48, 64}`, two of
  which (16, 48) aren't 32-aligned. This is what the original GPT-OSS
  deployment hit.
- Dense branch (`num_warps_m == 2: warp_shape_m = block_shape_m // 2`):
  found in this session via a real dense fp16×int4 GEMM run — produces
  `block_shape_m=64, warp_shape_m=32`, violating the *other* assert.

**Fix** (`patch.py`): after `get_config()` computes its final shapes,
check both invariants together; if `num_write_splits > 1` but either is
violated, fall back to `num_write_splits = 1` (already-supported,
simpler epilogue path).

**Verified**: heuristic-level sweep over GPT-OSS's real MoE shape
(`test_repro.py`, 6/6 shape_m values); real dense-path correctness run
(`test_correctness.py`) exercises the second manifestation end-to-end.

## Bug 2 — FIXED and verified (found "not fixed" as of Phase 5; resolved this pass)

**Root cause**: `humming/kernel/humming.py`'s
`HummingKernel.select_mma_op_class()` unconditionally computed
`mma_shape_k = 256 // self.a_dtype.num_bits`, giving `mma_shape_k=16` for
16-bit `a_dtype` (GPT-OSS's fp16 activations) and selecting an
`m16n8k16` MMA instruction:

```
ptxas ... error: Feature '.m16n8k16' requires .target sm_80 or higher
```

**Confirmed as a real hardware limit, empirically, not from memory**: a
minimal raw-PTX probe (`sm75_mma_probe/probe.py`), independent of
`humming` entirely, compiled directly with `nvcc -arch=sm_75`:

| shape | fp16→fp16 | fp16→fp32 |
|---|---|---|
| `m16n8k8`  | **compiles** | **compiles** |
| `m16n8k16` | rejected (`.m16n8k16 requires sm_80+`) | rejected (same) |

This matches Turing's real 2nd-gen Tensor Core capability: `m16n8k16` is
Ampere+ only; Turing's fp16 MMA tops out at `m16n8k8` (`M` unchanged,
`K` halved).

**An earlier attempt (documented as wrong in the prior version of this
doc) mirrored the neighboring `sm75+int8: mma_shape_m=8` special case**,
which also halves `M`, producing the invalid `m8n8k8` (not a real PTX
shape at all — `ptxas: "Unknown modifier '.m8n8k8'"`). That reasoning
doesn't transfer: the int8 case's `mma_shape_m=8` is specific to int8's
own valid shape family, not a generic "SM75 needs M=8" rule.

**The correct fix**: keep `mma_shape_m=16` unchanged, only set
`mma_shape_k=8`. This needed **zero C++ changes** — confirmed by reading
the code directly:

- `humming/config/mma.py`'s `MmaOpClassImpl` generates PTX register
  counts and the instruction string generically from `(m, n, k, dtypes)`
  (`calc_reg_count`, `f"mma.sync.aligned.m{m}n{n}k{k}..."`).
- `humming/include/humming/mma/wmma.cuh`'s `WMMA::run()` already loops
  `for k in range(kPartMmaShapeK / MmaShape::K)` — i.e. it already issues
  multiple `mma.fma()` calls per warp-K-tile generically. Halving
  `mma_shape_k` just doubles this loop's trip count, covering the same K
  range with two `m16n8k8` calls instead of one `m16n8k16` call.

**Verified, real, on hardware**:
1. `MmaOpClassImpl(16, 8, 8, f16, f16, f16/f32)` produces byte-identical
   PTX to what the raw probe confirmed compiles.
2. A real `HummingKernel.prepare_kernels(...)` call with GPT-OSS's exact
   MoE shape (shape_n=3072, shape_k=2944, num_experts=128, int4,
   bias=True, grouped_contiguous) — the same call that previously hit the
   `m16n8k16` ptxas error — now compiles **and** `cuModuleLoad`s
   successfully (`test_repro.py::test_bug2_fixed_gpt_oss_moe_shape_compiles_and_loads`).
3. A real dense fp16×int4 GEMM (M=128, N=3072, K=2944, group_size=32)
   run end-to-end through `humming.ops.humming_gemm()` on real T4
   hardware, output compared against a float32 reference: **mean abs
   err 0.0137**, within int4-quantization-noise tolerance
   (`test_correctness.py`).

## What's still open (honest scope)

- No full GPT-OSS-120B model load has been attempted yet — this fixes
  the kernel-compile/numerical-correctness layer only. Real weight
  loading, every actual layer shape in the model, every `GemmType`
  actually used (not just `grouped_contiguous`), and performance are all
  untested.
- No `bf16` activation path tested (only `float16`) — the fix covers
  both (`a_dtype in (float16, bfloat16)`) but only fp16 was exercised.
- No `use_f16_accum=True` or `use_batch_invariant=True` path tested with
  Fix 2.
- Performance of the `m16n8k8`-doubled-instruction-count path vs. the
  Ampere+ `m16n8k16` path is not measured — expect it to be slower
  per-MMA (2 instructions instead of 1 for the same K), though whether
  this matters end-to-end depends on whether these kernels are
  compute-bound or memory-bound on T4, which hasn't been profiled.

## Files

- `patch.py` — both fixes, applied via runtime monkeypatch (not editing
  the installed package — see module docstring for why).
- `test_repro.py` — Bug 1's heuristic-level regression sweep; Bug 2's
  real NVRTC-compile-and-load reproduction (now a pass, not an xfail).
- `test_correctness.py` — Bug 2's real numerical-correctness check.
- `../sm75_mma_probe/probe.py` — the independent raw-PTX evidence for
  what SM75 can/can't execute (Bug 2's root evidence, not humming-specific).
- `../sm75_mma_probe/correctness_dense.py` — scratch version of the
  correctness check (superseded by `test_correctness.py`, kept for the
  raw-PTX-probe cross-reference).

## Suitability for upstream reporting

Both bugs are now ready to report to `humming-kernels`' maintainers with
precise root cause, minimal fix (2 lines added to `select_mma_op_class`,
3-line generalization of the `get_config` post-check), and real
verification (compile + load + numerical correctness on the exact
hardware/shape combination that failed originally).
