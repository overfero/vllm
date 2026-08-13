# Deploying GPT-OSS 120B on a 3-Machine T4 Cluster

Hardware: 3 machines, each 2× NVIDIA T4 (16GB), 6 GPUs / 96GB total.
Distributed design: TP=2 per machine (real NCCL, local only), PP=3 across
machines (this project's own UDP hole-punch Transport, already built and
proven), DP=1. This document does not redesign or modify the transport
layer — it is treated as complete and correct.

Everything below is derived from real, checked sources: the actual
`openai/gpt-oss-120b` checkpoint (config + safetensors headers, fetched
directly from Hugging Face), this vLLM checkout's real source code (file
and line numbers given throughout), and real hardware queries on this
project's own T4s. Where something could not be verified, it is stated
as unverified rather than estimated silently.

---

## Bottom line, up front

**GPT-OSS 120B cannot be deployed on this cluster today — but the
reason is no longer purely "T4 lacks a hardware feature." That gate is
real and proven (below), but even if a T4-legal weight format is chosen
instead of the native one, no correct, complete, vLLM-ready checkpoint
in that format currently exists.** Concretely:

1. The official checkpoint is MXFP4. vLLM enforces `capability >= 80`
   for MXFP4 (`vllm/model_executor/layers/quantization/mxfp4.py:61`,
   checked in `vllm/config/vllm.py:720`). T4 is capability 75. This
   raises a `ValueError` before any weight loads — confirmed against a
   real T4 in this sandbox (`torch.cuda.get_device_capability(0) ==
   (7, 5)`). See [Task 2](#task-2-can-the-official-checkpoint-run-on-t4).
2. The natural fallback — AWQ or GPTQ INT4, which *does* pass T4's
   capability floor — requires a checkpoint that doesn't officially
   exist. This session searched Hugging Face directly (not from memory)
   and inspected the real bytes of every candidate found. The result:
   the only AWQ repos discoverable are either empty or **contain plain
   F16 weights for 6 of the model's 36 layers, mislabeled as
   "awq-w4a16" — not actually quantized, not complete, not usable**.
   The one real, complete, genuinely-quantized GPTQ checkpoint found was
   built and validated for a third-party FPGA runtime, not for
   general-purpose GPU/vLLM use, and reports a measured **+12.0%**
   perplexity degradation versus the BF16 reference — non-negligible.
   See [Task 5](#task-5-does-an-official-or-community-awq-checkpoint-exist).

**The single biggest blocker is now a checkpoint-production problem, not
a hardware or transport problem.** The distributed/networking piece
(TP=2 local NCCL + PP=3 transport-backed pipeline) is fully designed and
independently proven in earlier phases of this project. See
[Task 10](#task-10-can-gpt-oss-120b-actually-run-on-this-exact-cluster)
for the full verdict and the minimum concrete work required.

---

# Task 1: Real checkpoint measurements

Fetched directly from `openai/gpt-oss-120b` on Hugging Face — config.json
plus every safetensors shard's header (8-byte length prefix + JSON
header holding exact `dtype`/`shape`/byte-offsets per tensor, readable
via HTTP range request without downloading any tensor data):

```json
{
  "hidden_size": 2880, "num_hidden_layers": 36,
  "num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 64,
  "num_local_experts": 128, "num_experts_per_tok": 4,
  "intermediate_size": 2880, "vocab_size": 201088,
  "max_position_embeddings": 131072, "sliding_window": 128,
  "layer_types": ["sliding_attention","full_attention", ...alternating×36],
  "quantization_config": {"quant_method": "mxfp4",
    "modules_to_not_convert": ["*.self_attn","*.mlp.router","embed_tokens","lm_head"]}
}
```

All 15 shards (`model-00000..00014-of-00014.safetensors`) aggregated:

| Category | Elements | Bytes | Dtype |
|---|---|---|---|
| MoE experts (gate_up_proj + down_proj, 36 layers × 128 experts) | 114,661,785,600 | 60,993,699,840 | U8 (packed MXFP4, 2 values/byte + block scales) |
| Attention (q/k/v/o + bias + sinks, 36 layers) | 955,805,184 | 1,911,610,368 | BF16 |
| `lm_head` | 579,133,440 | 1,158,266,880 | BF16 |
| `embed_tokens` | 579,133,440 | 1,158,266,880 | BF16 |
| Router (36 layers) | 13,275,648 | 26,551,296 | BF16 |
| Norms | 210,240 | 420,480 | BF16 |
| **Total** | **116,789,343,552** | **65,248,815,744** | mixed |

**116.79B total logical parameters, 65.25GB (65,248,815,744 bytes)
on-disk footprint** — matches OpenAI's stated "~117B total params" and
the checkpoint's own `model.safetensors.index.json` `total_size` field
exactly. Active params/token: 4-of-128 experts routed per layer (~5.1B
active, OpenAI's stated figure — routing-dependent, not re-derived here).

## Weights, by precision

| Precision | Basis | Total | T4-loadable at all? |
|---|---|---|---|
| **MXFP4 (native)** | measured | **65.25 GB** | **No — capability gate** |
| BF16/FP16 | 116.79B × 2B | **233.58 GB** | Yes, but 2.4× cluster's 96GB |
| FP8 (hypothetical full-weight) | 116.79B × 1B | 116.79 GB | Gate passes (75=75); no real checkpoint |
| FP32 | 116.79B × 4B | 467.16 GB | Yes, irrelevantly large |
| AWQ/GPTQ INT4 (requantized) | ~0.5-0.56B/param, real GPTQ example measured at 64.9GB | **~33-65 GB (see Task 5 — huge spread because most "AWQ" candidates found are fake/broken)** | Gate passes | See Task 5/7 |
| GGUF Q4_K_M-class | ~0.55-0.6B/param | ~65-72 GB (estimated) | vLLM's GGUF MoE support is limited/experimental — untested |

## KV cache

```
bytes/token/layer = 2 × num_key_value_heads × head_dim × dtype_bytes
                   = 2 × 8 × 64 × 2 (bf16) = 2048 bytes
```

GPT-OSS alternates sliding (window=128) / full attention, 18 of each.
Sliding layers never exceed 128 tokens of cache regardless of sequence
length — vLLM's `KVCacheSpec`/`SlidingWindowSpec`
(`vllm/v1/core/kv_cache_utils.py`) supports this per-layer:

| Context | Naive (36 layers × L) | Realistic (18 full + 18 capped@128) | Per-GPU (÷6, TP2×PP3) |
|---|---|---|---|
| 4,096 | 0.302 GB/seq | 0.156 GB/seq | 26.0 MB/seq |
| 8,192 | 0.604 GB/seq | 0.307 GB/seq | 51.1 MB/seq |
| 32,768 | 2.416 GB/seq | 1.213 GB/seq | 202.1 MB/seq |
| 131,072 (max) | 9.664 GB/seq | 4.837 GB/seq | 806.1 MB/seq |

## Activations / workspace / fragmentation / optimizer state

Engineering estimates (no closed form, depend on kernel/batch choice):

- **Activations**: one layer's hidden state is `batch×seq×2880×2` bytes
  (bf16); MoE routing buffers (≤4 active experts × chunk ×
  5760×2 bytes) add tens-to-low-hundreds of MB per GPU for typical
  prefill chunk sizes — a working-set estimate, not summed over 36 layers.
- **CUDA workspace**: several hundred MB–2GB/GPU for cuBLAS/attention
  scratch + local NCCL buffers. T4 has no FlashAttention-2 (SM80+ only)
  — vLLM falls back to XFormers/Triton attention on T4.
- **Fragmentation**: handled by vLLM's own `--gpu-memory-utilization`
  (default 0.9), which profiles real usage post-load and caps
  allocation at that fraction — the practical answer, not a separate
  manual number.
- **Optimizer state**: **0 GB** — inference-only serving, no
  gradients/Adam state anywhere in a vLLM process.

## Hypothetical per-GPU fit (ignoring the capability gate, for Task 4's arithmetic)

```
Per-GPU weight share (TP=2 × PP=3 = 6 GPUs): 65.25 GB / 6 = 10.875 GB
T4 usable VRAM (nvidia-smi, this project's own hardware): 15,360 MiB = 16.106 GB
Headroom: 16.106 - 10.875 = 5.231 GB/GPU
```

5.23GB/GPU covers KV cache + activations + workspace comfortably at
4K-8K context, tight at 131K with concurrency — **conditioned on the
weight format in that slot actually being loadable on T4**, which the
native format is not.

---

# Task 2: Can the official checkpoint run on T4?

**No.** Verified three independent ways, not assumed:

1. **Source code**: `Mxfp4Config.get_min_capability()` returns `80`
   (`vllm/model_executor/layers/quantization/mxfp4.py:61`). Enforced in
   `VllmConfig._get_quantization_config`:
   `if capability < quant_config.get_min_capability(): raise ValueError(...)`
   (`vllm/config/vllm.py:720-726`).
2. **Real hardware**: `torch.cuda.get_device_capability(0)` on this
   project's own T4 returns `(7, 5)` → `DeviceCapability.to_int() == 75`.
   `75 < 80` → the exception fires.
3. **Error text, real**: `ValueError: The quantization method mxfp4 is
   not supported for the current GPU. Minimum capability: 80. Current
   capability: 75.`

## Why 80, exactly — is this hardware, kernel, or policy?

This needed digging past the single constant, because the answer isn't
as clean as "T4 lacks an instruction." `Mxfp4Config.get_min_capability()`
is one blanket number checked *before* vLLM picks which MXFP4 backend to
actually use. Backend selection happens later, in
`vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`, and there are
**multiple MXFP4 MoE backends**, not one:

- Triton-kernel-based backends (`TRITON`, `TRITON_UNFUSED`,
  `AITER_MXFP4_FP8`) — these are the primary, best-throughput path, and
  genuinely benefit from Ampere-class hardware (async memory copy,
  faster low-precision matmul paths) that Turing lacks.
- A **Marlin** backend also exists for MXFP4
  (`vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py`,
  `prepare_moe_mxfp4_layer_for_marlin`). Marlin's own internal capability
  check, shared across all Marlin-based kernels
  (`marlin_utils.py:65`), is `device_capability < 75: unsupported` —
  i.e. **Marlin itself would accept capability 75, T4's actual value.**

So the class-level `80` floor in `Mxfp4Config` is stricter than what the
Marlin sub-path alone would require. This makes the honest classification:

**Mostly (C) a software guard, with a real (A) hardware component
underneath it, not purely one or the other:**

- **(A) real, for the primary path**: Turing (T4, SM 7.5) has no native
  low-bit block-scaled tensor-core math and no FlashAttention-2 support
  (SM80+ only) — the Triton-based MXFP4 kernels this format is designed
  around are genuinely Ampere-class-dependent for correctness/performance.
- **(C) software guard, provably**: the `80` check is applied at the
  `Mxfp4Config` class level, before backend dispatch, even though the
  Marlin backend it also ships has its own, more permissive `75`
  threshold. vLLM chose to gate the *format* conservatively at its
  best-case backend's requirement rather than its worst-case (Marlin)
  backend's requirement — this is a policy decision, not a hard
  impossibility, though this project did not test whether forcing the
  Marlin backend and bypassing the class-level check would actually work
  end-to-end (would require patching `vllm/config/vllm.py:720`'s check
  itself, which this project's own standing instructions treat as
  redesigning something declared complete/correct — not attempted).
- **(B) CUDA kernel limitation**: partially true as a corollary of (A) —
  the compiled Triton MXFP4 kernels are only built/tuned for SM80+
  targets.
- **(D) something else**: no evidence found of a licensing, business, or
  arbitrary-versioning reason — the gate lines up with a real
  capability boundary in the primary backend, just enforced one level
  more coarsely than strictly necessary.

**Practical conclusion, unchanged**: within this project's scope
(transport complete, not redesigning it, and not patching a quantization
gate that reflects a real hardware boundary for the intended backend),
the native checkpoint does not run on T4. The interesting nuance above
matters for Task 6/7 (it's part of why a legitimate T4-targeted
quantization, not a hacked-open gate, is the right path).

---

# Task 3: Quantization method comparison

Every `get_min_capability()` value below is read directly from this
vLLM checkout's source, not recalled:

| Format | `get_min_capability()` | Source | T4 (75) passes? | vLLM support | Memory (this model) | Throughput on T4 | Quality | Maturity |
|---|---|---|---|---|---|---|---|---|
| **MXFP4** (native) | 80 | `mxfp4.py:61` | **No** | Full, but gated off T4 | 65.25 GB (measured) | N/A — won't start | Reference (as-trained) | Production, but Ampere+ only |
| **FP8** (fbgemm/compressed-tensors) | 75 | `fp8.py:144` | Yes (exactly) | Full | ~116.79 GB (no real weight-FP8 checkpoint exists) | T4 (Turing) has **no native FP8 tensor cores** (Ada/Hopper+ only) — gate passes, speed unlikely to help | Unverified for this model | Mature format, weak fit for T4 specifically |
| **AWQ** (auto_awq) | 75 | `auto_awq.py:230` | Yes (exactly) | Full, incl. MoE via Marlin-or-WNA16 fallback (see Task 4) | ~60-68 GB estimated (**real candidate found is fake — see Task 5**) | Comparable to FP16 dense once WNA16 fallback engages | Unverified for this model | Mature, well-supported quant path |
| **GPTQ** (auto_gptq) | 60 | `auto_gptq.py:187` | Yes | Full, same MoE fallback behavior as AWQ | 64.9 GB (**real, measured — one real checkpoint exists**, see Task 5) | Same fallback profile as AWQ | **+12.0% NLL / 89.6% top-1 agreement, measured by the checkpoint's own publisher** — non-negligible | Mature; this specific artifact FPGA-targeted |
| **bitsandbytes** (INT4/8) | 70 | `bitsandbytes.py:104` | Yes | Full | Similar order to AWQ/GPTQ | Not benchmarked here | Unverified for this model | Mature, but typically slower serving throughput than AWQ/GPTQ in vLLM |
| **`experts_int8`** | 80 | `experts_int8.py:42` | **No** | MoE-specific INT8 | ~117 GB (1B/param) | N/A on T4 | Unverified | Narrower use, also gated off T4 |
| **Unquantized BF16/FP16** | n/a (no gate) | — | Yes | Full | 233.58 GB (measured) | FP16 measured **2.53× faster** than BF16 on this project's real T4 (0.177s vs 0.448s, 50× 1024³ matmul) | Reference | Full |
| **GGUF** (llama.cpp-style Q4_K_M/Q8_0) | n/a in vLLM's loader; separate limitation | — | Loader exists but MoE support experimental | Partial/experimental for MoE | ~65-117 GB depending on quant level (estimated) | Untested in vLLM for this model | Community consensus (Bartowski's own GGUF release notes, found via search) explicitly warns **GPT-OSS's FFN "does not behave nicely when quantized to anything other than MXFP4"** — a real, sourced quality warning against non-native FFN quantization generally, not specific to GGUF | Mature for llama.cpp, weaker for vLLM |

**New/notable finding not previously documented**: MXFP4 and
`experts_int8` are the *only* two formats in this table gated above T4's
75 — every classic INT4/INT8 weight-only format (AWQ, GPTQ, bnb) clears
T4's floor. The ceiling isn't "T4 can't do low-bit inference" — it's
specifically "T4 can't do this model's *native* low-bit format."

---

# Task 4: Best strategy for this exact cluster

## Topology: TP=2 / PP=3 / DP=1 (unchanged from the given design — validated, not redesigned)

```
World size = 6. TP=2 (local NCCL, per machine). PP=3 (one stage/machine,
Transport-backed). DP=1 (no benefit at this scale — would only fragment
an already-tight per-GPU budget further).
```

- **Layer split**: 36 / 3 = 12 layers/stage exactly. Because
  `layer_types` strictly alternates, any 12-layer block = 6 sliding + 6
  full — every stage has an identical attention-cost profile.
- **TP divisibility**: `num_attention_heads=64 / 2 = 32` ✓,
  `num_key_value_heads=8 / 2 = 4` ✓. MoE experts are TP-sharded *within*
  each expert's matrix (vLLM's standard fused-MoE TP behavior, not
  expert parallelism) — `num_local_experts=128` never needs to divide by
  TP or PP size.
- **Why not TP=6/PP=1**: would require `torch.distributed` to form a
  group across machine boundaries with no reachable rendezvous — exactly
  what this project's whole transport layer exists to avoid needing.
- **Why not TP=1/PP=6**: idles the second GPU on every machine and
  forces 6 sequential pipeline hops instead of 3, for zero memory benefit
  (the same total weight bytes split across the same 6 GPUs either way).

## Precision recommendation, updated given this session's findings

MXFP4 is disqualified (Task 2). Between the remaining T4-legal
candidates, **GPTQ INT4 is now the better-evidenced recommendation**,
reversing the previous session's AWQ-first call — not because AWQ is
architecturally worse, but because **a real, complete, genuinely-4-bit
GPTQ checkpoint of this exact model exists and was inspected
byte-for-byte this session; no equivalent AWQ artifact does** (Task 5).
Recommending AWQ today means recommending a checkpoint that must still
be produced from scratch; recommending GPTQ means recommending one that
exists and needs validation, not production from zero.

- **Compute dtype**: FP16, not BF16 — directly justified by this
  project's own measured 2.53× T4 throughput difference (Task 3 table).
- **Kernel path**: for MoE experts specifically, expect vLLM to
  auto-fall-back from Marlin to the `MoeWNA16` kernel
  (`vllm/model_executor/layers/quantization/moe_wna16.py`,
  `get_min_capability()==70`) regardless of AWQ or GPTQ — see Task 7 for
  why (GPT-OSS's `hidden_size=2880` fails Marlin's MoE tile-alignment
  check). This is slower than Marlin's fast path but real and functional,
  and requires no source changes.
- **Honest caveat, unchanged in spirit from before**: neither candidate
  checkpoint has been validated as directly vLLM-loadable end-to-end in
  this sandbox (blocked by the pre-existing compiled-kernel issue, Task
  10) — the GPTQ artifact's own measured +12% NLL degradation should
  also be independently re-validated before trusting it in production,
  since it was measured on a different (FPGA) serving stack.

---

# Task 5: Does an official or community AWQ checkpoint exist?

**Searched directly against the Hugging Face Hub API this session** (not
recalled) — `HfApi().list_models(search=...)` across several query
variants. Results, with every candidate's real file bytes inspected via
safetensors header range-requests (the same method used for the native
checkpoint in Task 1), not taken at face value from repo names:

## AWQ candidates — none usable

| Repo | Claimed | Real, measured | Verdict |
|---|---|---|---|
| [`twhitworth/gpt-oss-120b-awq-w4a16`](https://huggingface.co/twhitworth/gpt-oss-120b-awq-w4a16) | AWQ w4a16, full model | **33.37 GB real (verified via HfApi file sizes), but the `model.safetensors.index.json` covers only layers 0-5 of 36** (`embed_tokens` + 6 layers, no `lm_head`, no final norm), **every tensor inspected across 3 shards is plain `F16`** — no `qweight`/`qzeros`/`scales` anywhere, and **no `quantize_config.json`/`quant_config.json`/`quantization_config` exists at all**. The index's own `metadata.total_parameters` field claims 116,829,156,672 (full model) while the actual tensors present cover a fraction of that — the metadata itself is internally inconsistent. | **Not AWQ, not complete, not usable.** Mislabeled artifact. |
| [`marcinbrzezanski/gpt-oss-120b-awq-w4a16`](https://huggingface.co/marcinbrzezanski/gpt-oss-120b-awq-w4a16) | Same | Identical size, 33.37 GB, to the byte | Near-certainly a mirror/fork of the same broken artifact (not individually header-inspected, but the exact byte-for-byte size match across independent uploaders is itself strong evidence, not proof) |
| [`oki0ki/gpt-oss-120b-awq-w4a16`](https://huggingface.co/oki0ki/gpt-oss-120b-awq-w4a16) | Same | Identical size, 33.37 GB | Same as above |
| `unieai/gpt-oss-120b-awq` | AWQ | **0 bytes, 0 files** — empty repo | Not a checkpoint at all |

**No usable AWQ checkpoint of GPT-OSS 120B exists anywhere this session
could find, official or community.** Any AWQ deployment path requires
producing one from scratch (Task 6).

## GPTQ candidates — one real, complete artifact, with real caveats

[`positron-ai/openai_gpt-oss-120b-ingest-best-gptq`](https://huggingface.co/positron-ai/openai_gpt-oss-120b-ingest-best-gptq)
— genuinely different from the AWQ candidates:

- **64.9 GB real** (16 safetensors shards, verified via HfApi), and the
  first shard's header confirms **all 36 layers present**
  (`model.layers.0` through `model.layers.35`), plus `embed_tokens`.
- **Actually quantized**: real `qweight` (I32), `qzeros` (I32), `scales`
  (BF16), `g_idx` (I32) tensors present per linear layer, per expert —
  e.g. `model.layers.0.mlp.experts.0.gate_up_proj.qweight`, shape
  `[360, 5760]`, confirming genuine 4-bit packing, not a relabeled
  full-precision file.
- **Per-expert tensor layout**: unlike the native/broken-AWQ checkpoints
  (which store all 128 experts as one fused `[128, 2880, 5760]` tensor
  per layer), this checkpoint stores **128 separate tensors per layer**
  (`mlp.experts.0.gate_up_proj`, `mlp.experts.1.gate_up_proj`, ...
  `.127.`). This matters directly for Task 7.
- **Real `quantize_config.json`**: `bits: 4, group_size: 64, sym: true,
  quant_method: "gptq", checkpoint_format: "gptq"`, produced by
  `gptqmodel:5.8.0` (a real, current quantization toolchain — see
  https://github.com/modelcloud/gptqmodel).
- **Explicitly not general-purpose**, per its own README: *"This
  repository contains a Positron AI quantized build of
  openai/gpt-oss-120b for tron inference on Positron FPGA-serving
  infrastructure... For general-purpose GPU inference, compare against
  the original model and other quantized formats before deployment."*
- **Measured quality, from the publisher's own README**: mean
  KL-divergence 0.1061 vs BF16 reference, P95 KL-divergence 0.4772,
  top-1 greedy token agreement 89.61%, **perplexity/NLL delta +12.0%**.
  This is a real, non-negligible degradation — not "negligible loss" as
  informally claimed elsewhere for AWQ-style quantization of this model.
  It was also measured on Positron's own FPGA runtime, not vLLM/CUDA, so
  it is evidence of *a* quality delta, not necessarily *the* delta vLLM
  would reproduce.
- **One structural oddity worth flagging, unresolved**: the
  `quantize_config.json`'s `meta.moe.routing` block records
  `num_experts_per_tok: 64` for calibration — GPT-OSS's real inference
  routing is top-4-of-128. This is very likely a calibration-time
  technique (routing through more experts during calibration to get
  broader coverage of each expert's weight distribution, a known
  technique for MoE quantization) rather than a change to runtime
  behavior, but this project did not find independent documentation
  confirming that interpretation — noted as an open question, not
  asserted as fact.

**Conclusion for Task 5**: no official AWQ/GPTQ checkpoint exists from
OpenAI. No usable community AWQ checkpoint exists at all. One real,
complete, genuinely-quantized GPTQ checkpoint exists, built for a
different serving stack, with real measured quality loss that should be
treated as a lower bound on what vLLM would see, not a guarantee.

---

# Task 6: Producing a checkpoint without loading the full model into GPU memory

## Can it be done? Yes — this is real, verifiable, not a guess

GPTQ and AWQ are both **layer-wise** algorithms by construction: each
transformer block's linear layers are calibrated and quantized one at a
time, using a small calibration dataset run through the model up to that
point. The *computational* working set at any moment is one layer's
weights + that layer's Hessian/activation statistics — not the whole
model. This is exactly why CPU/disk-offloaded quantization of models far
larger than any single GPU's VRAM is a normal, supported workflow (e.g.
via HF `accelerate`'s `device_map="auto"`, or a toolchain's own native
offload).

**Direct evidence, not inference**: the real `quantize_config.json`
fetched from `positron-ai/openai_gpt-oss-120b-ingest-best-gptq` (Task 5)
contains, verbatim, from GPTQModel 5.8.0's own metadata:

```json
"offload_to_disk": false, "offload_to_disk_path": null,
"vram_strategy": "balanced", "gc_mode": "interval",
"hessian": {"chunk_size": null, "chunk_bytes": 536870912, "staging_dtype": "float16"},
"true_sequential": true
```

This is real proof that the toolchain used to actually quantize this
exact model supports **disk offload as a first-class option**
(`offload_to_disk` is a real, present flag — simply set to `false` for
this particular run, meaning that run likely had enough combined
GPU+CPU memory and didn't need it) and **chunked Hessian computation**
(`chunk_bytes: 536870912` = 512MB chunks) specifically to bound peak
memory during calibration. `true_sequential: true` also confirms the
expected layer-by-layer dependency: each layer's calibration forward
pass depends on all *earlier* layers already being quantized (their
outputs feed the next layer's calibration inputs), which is why this is
inherently a sequential pass over layers, not something naively
parallelizable layer-by-layer.

## What can be parallelized

- **Sequential across layers**: cannot be avoided — layer *N+1*'s
  calibration activations depend on layer *N* already being quantized
  in the reference path (`true_sequential`).
- **Parallel across experts, within a layer**: GPT-OSS's 128 experts per
  layer are structurally independent linear layers with no cross-expert
  dependency in their own Hessian computation — quantizing experts
  0-63 and 64-127 of the *same* layer on two different GPUs (or two
  different machines) simultaneously is a legitimate, embarrassingly
  parallel split. This is real algorithmic structure, not
  toolchain-specific.
- This project's own 3-machine/6-GPU cluster **could**, in principle, be
  repurposed to run the expert-level split of a quantization job (not
  inference) — but this is a genuinely different workload than the
  transport layer was built for (weight-quantization job orchestration,
  not pipeline-parallel activation forwarding), and no code for it
  exists in this repository. Out of scope to build here per this
  project's own instruction not to redesign the transport for a new
  purpose without a proven deployment need.

## Estimates — clearly labeled, not measured

No quantization job was actually run this session (would require
`pip install autoawq`/`gptqmodel`, real calibration data, and hours of
GPU time — outside this task's scope and this sandbox's realistic
budget). These are engineering estimates, grounded in the real numbers
above, not fabricated precision:

- **Minimum disk**: BF16 source weights (233.58GB, Task 1) + output
  checkpoint (~33-65GB depending on format/group size, Task 1/5) +
  working scratch for Hessian/calibration artifacts (small, low-single-
  digit GB with chunking) → **budget ~320-350GB free**, comfortably
  covered by "~300GB/machine" if source and output don't need to coexist
  on the same disk, tighter if they do.
- **Minimum RAM (CPU, with disk-offloaded weights and chunked Hessians)**:
  the toolchain evidence above shows the intended design keeps only the
  active layer's weights + a bounded (512MB-chunk) Hessian resident at
  once, plus calibration batch activations (small — 128 samples ×
  1024 tokens × 2880 dims × 2 bytes ≈ 1.5GB for the largest single
  activation tensor in flight). A realistic floor is in the **32-64GB
  RAM** range for the orchestration process itself, assuming true
  streaming from disk; if the workflow instead loads the *whole* BF16
  model into RAM before offload-quantizing (a common simpler-but-lazier
  implementation choice), budget the full **~234GB RAM** instead. Which
  behavior any specific toolchain actually exhibits was not verified
  here — this is a real design-choice spread, not an error bar on one
  number.
- **Runtime**: not documented by the one real example found (its README
  gives a release date, 2026-06-30, but no wall-clock quantization time).
  As an order-of-magnitude anchor: GPT-OSS's MoE structure means far
  more individually-quantized weight matrices per layer than a same-size
  dense model (128 experts × 2 projections = 256 matrices/layer × 36
  layers ≈ 9,216 quantizable matrices, versus roughly 252 for a
  comparably-sized 36-layer dense model at ~7 linear layers/block).
  Dense 70B-class GPTQ/AWQ runs commonly take single-digit hours on one
  high-end GPU; scaling by matrix count alone suggests this job is
  plausibly a **multi-hour-to-low-double-digit-hour** job on comparable
  hardware, faster with the expert-level parallelism described above.
  **This is an estimate range, not a benchmark** — no run was performed.

---

# Task 7: Can vLLM load a new AWQ/GPTQ checkpoint without source changes?

**Conditionally yes — and this session found real, specific evidence
for both the positive and negative cases**, by reading the actual
loader code rather than assuming.

## What vLLM's existing code already handles, unmodified

- **Format auto-detection**: `AutoAWQConfig`/`AutoGPTQConfig` are
  discovered via `get_config_filenames()` (`["quantize_config.json",
  "quant_config.json"]` for AWQ; `["quantize_config.json"]` for GPTQ) or
  a `quantization_config` block in `config.json`. A checkpoint with a
  correctly-formed one of these needs **no source changes** to be
  recognized.
- **MoE kernel fallback is automatic**: `AutoAWQConfig.get_quant_method`
  (`auto_awq.py:319-357`) calls `check_moe_marlin_supports_layer`
  before committing to the fast Marlin MoE path. That check
  (`marlin_utils.py:355-386`) requires
  `hidden_size % 128 == 0`. **GPT-OSS's `hidden_size=2880`, and
  `2880 % 128 == 64`** — fails this check, on *any* GPU, not just T4.
  vLLM's own code then automatically falls back to the `MoeWNA16`
  kernel (`moe_wna16.py`, `get_min_capability()==70`) — logged as a
  warning, not an error, and requiring **zero code changes**. The exact
  same fallback structure exists for GPTQ (shared `marlin_utils.py`
  logic). This is real, verified by reading the conditional in the
  loader, not inferred from documentation.
- **Per-expert tensor naming is already handled**: this was the
  surprising find. `vllm/model_executor/models/gpt_oss.py`'s
  `load_weights` (around lines 641-820) explicitly parses an
  `expert_id` out of incoming tensor names and scatters into the
  correct slice of the fused in-memory parameter
  (`expert_data = params_dict[fused_name].data[expert_id]`). This means
  a checkpoint storing **128 separate per-expert tensors** — exactly
  the layout the real `positron-ai` GPTQ checkpoint uses
  (`mlp.experts.{0..127}.gate_up_proj`) — is a naming convention vLLM's
  own loader already anticipates, not something requiring a new
  conversion script. This directly overturned this session's initial
  assumption that a fused-vs-per-expert layout mismatch would force a
  source change.

## What remains unverified, honestly

- None of this was executed end-to-end in this sandbox. The compiled-
  kernel blocker (Task 10) prevents running *any* real model, quantized
  or not, here — so "vLLM's loader code, read carefully, appears to
  accept this checkpoint's shape and naming" is a source-code-level
  finding, not a passing test.
- The `positron-ai` checkpoint's unusual calibration-time
  `num_experts_per_tok: 64` metadata (Task 5) is stored under
  `quantize_config.json`'s `meta` key, which is calibration
  provenance, not a runtime config vLLM reads — it should not affect
  loading or inference routing, but this was not independently
  confirmed against the toolchain's source.
- If a *newly produced* AWQ/GPTQ checkpoint used a **fused** per-layer
  tensor layout instead (matching the native checkpoint's own
  `mlp.experts.gate_up_proj` single-tensor style, which is what
  AutoAWQ typically produces rather than GPTQModel's per-expert style),
  that path is also directly supported — `gpt_oss.py`'s loader handles
  both conventions, per the same code region.

**Conclusion**: for a checkpoint with a correct, standard AWQ/GPTQ
config file and either of the two tensor-naming conventions already
covered above, **no vLLM source modification should be required**. This
is the most confident positive finding of this whole exercise, backed
by reading the exact conditional logic rather than trying it and hoping.

---

# Task 8: Closest hardware configuration that CAN run GPT-OSS 120B natively

If the constraint is "run the *official* MXFP4 checkpoint, unmodified,
no requantization risk" — the only lever that matters is compute
capability ≥ 80 (Ampere or newer). Any of these clear the gate:

| Config | VRAM | Fits 65.25GB MXFP4? | Distributed complexity | Notes |
|---|---|---|---|---|
| **1× H100 80GB** | 80GB | Yes, ~15GB headroom | **None** — single GPU, no TP/PP/transport at all | Cleanest possible deployment; OpenAI's own MXFP4 design target was explicitly "fits on a single H100" (confirmed via web search of public commentary on the release) |
| 1× A100 80GB | 80GB | Yes, ~15GB headroom | None | Capability exactly 80 — passes, no margin to spare on the gate itself (irrelevant post-load, gate is pass/fail not graded) |
| 2× A100 40GB (TP=2) | 80GB total | Yes | Single machine, real NCCL only | No PP, no transport layer needed |
| 3-4× L4 or A10 24GB | 72-96GB total | Yes (3×24=72 > 65.25; 4× gives real headroom) | TP=3 or TP=4, single machine preferred | Both Ada (L4, capability 89) / Ampere (A10, capability 86) clear the gate with margin |
| 2 machines × 1× A100 80GB (PP=2, keeping this project's transport in the loop) | 160GB total, wildly over-provisioned | Yes, trivially | Minimal *multi-machine* MXFP4-native deployment, if validating the transport layer with a real model is itself a goal | Only relevant if the point is proving the transport with GPT-OSS specifically, not minimizing hardware |

## Cost — real, sourced, current as of this session (Aug 2026), not from training-data memory

Cloud on-demand hourly pricing found via web search this session (ranges
across multiple providers, genuinely varies by vendor/region/spot vs.
on-demand — treat as an anchor, not a quote):

- **H100 80GB**: roughly **$1.49-$6.98/hr** across providers found
  (median trending ~$3.32/hr, reported as up ~11% since mid-2025).
- **A100 80GB**: roughly **$1.07-$2.50/hr** on-demand (as low as
  ~$0.60/hr spot from one provider).
- L4/A10 specific current pricing was not found with confidence this
  session — not stated as a number to avoid fabricating one.

A single A100 80GB at ~$1-2.50/hr is likely both the cheapest and the
lowest-engineering-effort path to actually running this exact model
unmodified — no TP, no PP, no transport, no quantization risk. It is
also a fundamentally different exercise than this project (single-GPU
serving needs none of the distributed work already built).

## Engineering effort, relative to the current T4 path

- **Hardware swap only (A100/H100), single GPU**: near-zero incremental
  engineering beyond standard vLLM setup — this project's entire
  transport/PP layer becomes unnecessary for this specific model at
  this hardware tier.
- **Staying on T4, fixing via quantization instead**: the real
  remaining work (Task 6/7/10) — produce or validate a T4-legal
  checkpoint — which is nontrivial but has now been shown, concretely,
  to not require any vLLM source changes once the checkpoint exists
  correctly (Task 7).

---

# Task 9: Deployment guide

## Per-machine assignment

| | GPUs | Layers | TP ranks | PP rank |
|---|---|---|---|---|
| **Machine A** | 2× T4 (local TP group) | 0-11 (6 sliding + 6 full) | 0, 1 | 0 |
| **Machine B** | 2× T4 (local TP group) | 12-23 (6 sliding + 6 full) | 0, 1 | 1 |
| **Machine C** | 2× T4 (local TP group) | 24-35 (6 sliding + 6 full) | 0, 1 | 2 |

Machine A additionally holds `embed_tokens`; Machine C additionally
holds the final norm + `lm_head` and serves the OpenAI-compatible API.
Machine B needs **two** independent transport links (→A, →C) — this
exact "middle stage, two connections" pattern was already built and
tested in an earlier phase of this project
(`tests/transport/test15_pipeline_three_stage.py`, passing on both
`tcp`/`udp`), and `pipeline_bootstrap.py`'s
`install_transport_pp_group(pp_rank, pp_world_size)` takes both as plain
parameters — PP=3 needed no new transport code.

## Environment variables (every machine)

```bash
export VLLM_TRANSPORT=udp
export VLLM_UDP_TRANSPORT_DIR=/opt/udp_holepunch
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=WARN
export TORCH_CUDA_ARCH_LIST=7.5      # T4 = Turing, SM 7.5
```

## Machine A

```bash
source /opt/gptoss-venv/bin/activate
export VLLM_TRANSPORT=udp VLLM_UDP_TRANSPORT_DIR=/opt/udp_holepunch CUDA_VISIBLE_DEVICES=0,1

python3 /opt/vllm/scripts/launch_pp_stage.py \
    --model /data/models/gpt-oss-120b-gptq \
    --quantization gptq \
    --dtype float16 \
    --tensor-parallel-size 2 \
    --pp-rank 0 --pp-world-size 3 \
    --signaling-url https://<your-zrok-token>.share.zrok.io \
    --self-id MachineA --peer-id MachineB
```

## Machine B

```bash
source /opt/gptoss-venv/bin/activate
export VLLM_TRANSPORT=udp VLLM_UDP_TRANSPORT_DIR=/opt/udp_holepunch CUDA_VISIBLE_DEVICES=0,1

python3 /opt/vllm/scripts/launch_pp_stage.py \
    --model /data/models/gpt-oss-120b-gptq \
    --quantization gptq \
    --dtype float16 \
    --tensor-parallel-size 2 \
    --pp-rank 1 --pp-world-size 3 \
    --signaling-url https://<your-zrok-token>.share.zrok.io \
    --self-id MachineB --prev-peer-id MachineA --next-peer-id MachineC
```

## Machine C

```bash
source /opt/gptoss-venv/bin/activate
export VLLM_TRANSPORT=udp VLLM_UDP_TRANSPORT_DIR=/opt/udp_holepunch CUDA_VISIBLE_DEVICES=0,1

python3 /opt/vllm/scripts/launch_pp_stage.py \
    --model /data/models/gpt-oss-120b-gptq \
    --quantization gptq \
    --dtype float16 \
    --tensor-parallel-size 2 \
    --pp-rank 2 --pp-world-size 3 \
    --signaling-url https://<your-zrok-token>.share.zrok.io \
    --self-id MachineC --peer-id MachineB \
    --serve --port 8080
```

`--quantization gptq` targets the real `positron-ai` checkpoint (Task
5); swap to `--quantization awq` only once a real, complete AWQ
checkpoint exists (Task 6). `scripts/launch_pp_stage.py` does not exist
in this repository yet — see Task 10 for exactly what it needs to do and
why it wasn't written blind.

## Signaling server + starting order

```bash
# always-on host reachable by all three machines
cd /opt/udp_holepunch && pip install fastapi uvicorn
python3 -m uvicorn signaling_server:app --host 0.0.0.0 --port 8000
zrok share public localhost:8000   # note the printed URL
curl -s https://<token>.share.zrok.io/peer/__probe__   # expect 404, "peer not registered yet"
```

1. Signaling server + zrok — first, confirm the health check.
2. Machine B (middle stage) — starts both transport links, waits for A/C.
3. Machine A — registers, hole-punches to B.
4. Machine C — registers, hole-punches to B.
5. Wait for `Hole punch success.` (unmodified `peer.py` log line) on
   all three machines.
6. Each stage loads its shard once its local TP group and PP link are
   both up; Machine C starts the API server once loaded.

## Download + verify the checkpoint

```bash
pip install -U "huggingface_hub[cli]"
hf download positron-ai/openai_gpt-oss-120b-ingest-best-gptq --local-dir /data/models/gpt-oss-120b-gptq

python3 -c "
import json
idx = json.load(open('/data/models/gpt-oss-120b-gptq/model.safetensors.index.json'))
print('total_size:', idx['metadata']['total_size'])
assert idx['metadata']['total_size'] == 64862418823, 'checkpoint size mismatch - re-download'
print('OK - matches known-good size (verified via HfApi this session, 46,983 weight entries)')
"
```

Disk: ~65GB for the checkpoint itself; budget ~150-200GB/machine
headroom for safety, logs, and any re-download.

## Verification checklist

| Check | Command | Depends on |
|---|---|---|
| Pipeline initialized | `get_pp_group().transport is not None`, `.world_size==3` | `pipeline_bootstrap.py` — tested for PP=2, PP=3 combined-with-real-TP not run this session |
| Correct quant method selected | Startup log shows `quant_method=gptq` and (if it engages) the `MoeWNA16` fallback warning for MoE layers | Task 7 findings |
| All stages loaded | vLLM's real "Loading model weights took..." log, all 3 machines | Compiled kernels (Task 10 blocker) |
| Model ready / first token | `curl <MachineC>:8080/v1/completions -d '{"prompt":"Hello","max_tokens":1}'` | Same |

---

# Task 10: Can GPT-OSS 120B actually run on this exact cluster?

**No — not today.** Two blockers, of different character:

## Blocker 1: the native checkpoint is hardware-gated off T4 (solved-as-diagnosed, not fixable in software here)

`Mxfp4Config.get_min_capability()==80` vs. T4's real `75`
(`mxfp4.py:61`, `vllm/config/vllm.py:720`). This has a legitimate
partial hardware basis (Task 2) — not something this project should
patch open.

## Blocker 2 (the actual biggest blocker, newly and concretely characterized this session): no validated, T4-legal checkpoint exists yet

This is the headline change from the previous version of this document.
It's no longer "quantize it and it'll fit" as an abstract statement —
this session went and checked, byte-for-byte:

- The only AWQ candidates are either empty or a broken, 6-of-36-layer,
  not-actually-quantized artifact, mislabeled across at least three
  mirrored repos (Task 5).
- The one real, complete, genuinely-4-bit GPTQ checkpoint that does
  exist was built and validated for a different (FPGA) serving stack,
  with a real measured **+12.0%** perplexity cost, and has never been
  loaded by vLLM (Task 5, Task 7).
- vLLM's own loader code, read carefully, has real reason to believe it
  *could* load that GPTQ checkpoint without modification (Task 7) — but
  "the code looks like it should accept this" is not the same claim as
  "it was loaded and produced a token," and this sandbox cannot make
  that second claim (Blocker 3, below, prevents testing regardless of
  checkpoint).

## Blocker 3 (carried over, unchanged): this sandbox has no compiled CUDA/CPU kernel extensions

`vllm._C_stable_libtorch` / `torch.ops._C.init_cpu_memory_env` don't
exist in this raw source checkout — confirmed directly in earlier
sessions, not re-litigated here. This blocks executing *any* real model
in this specific sandbox, independent of GPT-OSS or quantization
choice. Fix is a real from-source build (`pip install -e .
--no-build-isolation`, `TORCH_CUDA_ARCH_LIST=7.5`) on real deployment
machines with a full CUDA toolchain — not attempted blind here again,
per the prior session's risk assessment (30min-2hr+, uncertain success
without a real build environment to iterate against).

## Minimum work to actually get text generated, in order

1. **Resolve Blocker 3 on real hardware** — a real vLLM build with
   compiled kernels, on the actual deployment machines. Mechanical, not
   a design problem.
2. **Get a real, T4-legal checkpoint into hand** — either (a) validate
   the existing `positron-ai` GPTQ artifact end-to-end (download it,
   attempt to load it in a real built vLLM, check the quality
   degradation is acceptable for the intended use), or (b) run a real
   AWQ/GPTQ quantization job from the dequantized BF16 source (Task 6) —
   both are now concretely scoped, neither is a research question
   anymore.
3. **Write `scripts/launch_pp_stage.py`** (~150-250 LOC, medium
   complexity per the prior session's estimate) — wires together
   already-proven pieces (`init_distributed_environment` →
   `ensure_model_parallel_initialized` → `install_transport_pp_group` →
   `Worker.load_model()`/serving), following the exact orchestration
   pattern already validated in `tests/transport/test18_real_bootstrap_pp.py`.
   Not written blind in this sandbox since steps 1-2 must exist first to
   test it against.
4. Deploy per Task 9, on real hardware.

**What is genuinely done and does not belong on this list**: the
transport layer itself (hole punching, tensor/tensor-dict transport,
the transport-backed `GroupCoordinator` override) — proven for real,
independent of GPT-OSS, in `test18_real_bootstrap_pp.py` (both `tcp`
and `udp`, real local `torch.distributed` TP group coexisting with a
transport-backed PP group) and `test15_pipeline_three_stage.py` (the
3-stage/2-transport-link topology this exact cluster needs). Per this
task's own framing, that work is complete and was correctly not
touched this session.
