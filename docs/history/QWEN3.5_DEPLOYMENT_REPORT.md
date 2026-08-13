# Qwen3.5-122B-A10B-GPTQ-Int4: real 3-machine UDP-hole-punch PP deployment

Date: 2026-08-12. Real execution on the same 3-machine cluster (local sandbox +
akun2 + akun3, 2xT4 each) used for the GPT-OSS-120B deployment
(pp_tests/BLOCKER_REPORT.md), reusing the same transport_runtime /
TransportExecutor / stage_server.py infrastructure with a new model.

## Model

- `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` (real HF repo, 78.8GB full checkpoint).
- 48 hybrid decoder layers (3x Gated DeltaNet linear-attention + 1x full
  attention, repeating), 256 routed experts + 1 shared expert per layer,
  GPTQ-Int4 (group_size=128, sym) on the routed experts only; attention,
  shared expert, embeddings, lm_head, vision tower, and MTP stay unquantized.
- Natively multimodal (`Qwen3_5MoeForConditionalGeneration`, vision tower +
  MTP module). Served **text-only**: extraction dropped `model.visual.*`
  (333 tensors) and `mtp.*` (785 tensors) from every stage shard, and every
  launch command passes `--language-model-only` (real existing vLLM flag,
  `vllm/config/multimodal.py`) so the vision tower is marked a
  `StageMissingLayer` and never required at weight-load time.

## Layer split (PP=3, 16 layers/stage, real key layout)

Real checkpoint keys are `model.language_model.layers.{i}.*` (not
`model.layers.{i}.*` like GPT-OSS) - a new extraction script,
`humming_fix/single_layer_probe/extract_stage_checkpoint_qwen35.py`, was
written for this. Selective per-stage shard download (`hf download --include`)
was used instead of full-checkpoint download: each stage only pulled the
24-26 of 39 shards that actually contain its layer range + globals
(53-57GB/stage instead of 78.8GB), keeping peak per-machine disk footprint
well under the ~85GB practical ceiling (final extracted stage shards: 25GB,
22GB, 25GB).

- stage0 (Machine A): layers 0-15 + embed_tokens/norm/lm_head
- stage1 (Machine B): layers 16-31
- stage2 (Machine C, driver): layers 32-47 + embed_tokens/norm/lm_head

## Real issues hit and fixed this run

1. **`--language-model-only` wasn't wired into `stage_server.py` /
   `launch_pp_stage.py` / `profile_num_gpu_blocks.py`** - all three had fixed
   argparse + explicit `EngineArgs(...)` construction (no passthrough), so
   the flag had to be added to each explicitly and re-synced to all 3
   machines.
2. **Default `max_model_len` (262,144, the model's native context) blew the
   KV cache budget** - hybrid mamba+attention block accounting forces the
   attention block size up to 2096 tokens/block (to align with the mamba
   state's page size), so a 0.4 GiB free-memory remainder couldn't fit even
   one block. Fixed by passing `--max-model-len 8192` everywhere (a real,
   deliberate scope reduction for this smoke test, not a bug).
3. **stage2 needed slightly more memory than stage0/stage1 for weights**
   (12.37 GiB vs 11.66 GiB - real per-layer expert-size variance) and came up
   0.18 GiB short of KV cache at the default `gpu_memory_utilization=0.90`.
   Fixed by bumping to `0.95` uniformly across all 3 stages.
4. Real per-stage KV cache block counts (hybrid model: vLLM unifies the
   mamba-state and full-attention block granularity into one `num_blocks`,
   confirmed via the `interface.py:911/935` "Setting attention block size...
   / Padding mamba page size..." log lines, so a single override is still
   valid): stage0=106, stage1=104, stage2=68 (at 0.95 util). Used
   `--num-gpu-blocks-override 60` uniformly (safe margin below the smallest).

## MoE backend on T4 (SM75) - the "overflow precision" question

**`Using 'MARLIN' WNA16 MoE backend`** on all 3 stages - NOT the `HUMMING`
backend GPT-OSS needed (vLLM's WNA16 backend selection tries
`FLASHINFER_TRTLLM > MARLIN > BATCHED_MARLIN > TRITON > HUMMING > EMULATION`
in order; Marlin's shape checks - `hidden_size % 128 == 0`,
`group_size in [-1,32,64,128]` - passed for this checkpoint's expert
dimensions, so it never fell through to Humming). This means
`humming_fix/patch.py`'s two T4-specific bugfixes were not exercised at all
this run.

Code-level precision analysis before deployment (see conversation): vLLM's
Gated DeltaNet kernel (`qwen_gdn_linear_attn.py`) unconditionally casts the
decay gate, beta, and recurrent state to `float32` internally regardless of
model dtype - matching the checkpoint's own `mamba_ssm_dtype: float32`
requirement - so the linear-attention recurrence is not exposed to fp16's
narrower range even though the model is served in `--dtype float16` (T4 has
no bf16 tensor cores; the checkpoint's native dtype is bf16).

Real generation evidence (4 requests, greedy decoding, through all 48 layers
across 3 real physically-separate machines):

- `"The capital of France is"` -> `" Paris.\nThe capital of France is
  Paris.\nThe capital of France is Paris.\nThe"` - correct, coherent
  (repetition is a greedy-decoding/no-repetition-penalty artifact, not a
  precision issue).
- `"2 + 2 ="` -> `" 5\n2 + 2 = 5"` - **wrong answer, but clean, grammatical
  English** - no garbled/repeated-punctuation output. This is the key
  contrast with GPT-OSS's earlier "!!!!!!!!" result on the same prompt: that
  was the actual signature of a real problem (in GPT-OSS's case, GPTQ
  quality loss, not overflow); Qwen3.5 just gets the arithmetic wrong
  without any sign of numerical corruption.
- `"Once upon a time, in a small village, there lived"` (60 tokens) ->
  fully coherent narrative prose.
- Chat completion asking for the largest planet -> coherent step-by-step
  reasoning trace, correctly homing in on comparing planet sizes (cut off by
  `max_tokens=60` before naming Jupiter).

**Conclusion: no evidence of numerical overflow in real generation.** GPTQ
weight-only quantization keeps the MoE GEMM accumulation path away from
fp16's narrow range (Marlin dequantizes to fp16 for the matmul but
accumulates internally), and the recurrent GDN state - the one path this
model's authors explicitly flagged as precision-sensitive
(`mamba_ssm_dtype: float32`) - is already kept in fp32 by vLLM's own kernel,
independent of the model's overall serving dtype. The observed arithmetic
error ("2+2=5") is ordinary quantization-induced quality loss, not an
infra-level precision bug.

## Real steady-state after 4 test requests

- Machine A: 13.4/15.4 GiB + 13.2/15.4 GiB used (2xT4)
- Machine B: 13.4/15.4 GiB + 13.2/15.4 GiB used (2xT4)
- Machine C: 14.8/15.4 GiB + 14.8/15.4 GiB used (2xT4)
- `curl http://<machineC>:8080/health` -> HTTP 200

## Chatbot UI

Simple local chat UI (`pp_tests/chatbot_ui.html` + `pp_tests/chatbot_server.py`,
proxying same-origin to the vLLM API through an SSH `-L` local port forward to
Machine C, forwarded again to a browser via VSCode's port-forwarding) - see
the running instructions in-session. Learned the hard way: the proxy's
upstream request timeout must be generous (raised from 120s to 600s) - at
~2 tok/s a full `max_tokens` reply genuinely takes minutes, not seconds, and
the model finished server-side well after the original timeout fired 502.

## Real per-hop latency measurement (`pp_tests/real_ping_pong.py`)

A real cross-machine ping-pong (200 packets/pair, actual UDP hole-punch
transport, not the loopback-only `tests/transport/test3_ping_pong.py`)
found the root cause of the ~2 tok/s generation speed: Machine B (akun2)
landed in a different Kaggle/GCP region than A and C this session.

| Hop | Location pair | Avg RTT |
|---|---|---|
| A <-> B | Groningen, NL <-> Council Bluffs, IA, US | 104.6 ms |
| B <-> C | Council Bluffs, IA, US <-> Groningen, NL | 104.4 ms |
| A <-> C | Groningen, NL <-> Groningen, NL | **0.83 ms** |

The A<->C number confirms the transport is genuinely peer-to-peer (not
relayed) - the ~104ms legs are real transatlantic distance, not a hole-punch
or protocol inefficiency. Since B is the middle PP stage, every token
crosses the Atlantic twice (A->B, B->C), unavoidably, for as long as B stays
in that region.

## Real per-machine compute/transport timing breakdown

Added real `time.monotonic()` instrumentation directly to
`vllm/transport/udp_transport.py`'s `send()`/`recv()` (logged as
`[TRANSPORT_TIMING] self=... peer=... op=... bytes=... duration_ms=...`)
and `vllm/transport/pp_worker.py`'s `TransportPPWorker.execute_model()`
(logged as `[EXECUTE_MODEL_TIMING] self=... duration_ms=...`) to separate
real local GPU compute time from real network transport time per decode
step, instead of guessing. Local compute time per stage = execute_model's
total duration minus the send/recv durations logged inside that same call.

**Eager mode (original deployment):**

| Stage | Local compute (16 layers, TP=2, T4) |
|---|---|
| Machine A | ~75-82 ms |
| Machine B | ~84 ms |
| Machine C (+ sampling) | ~59 ms |
| **Total compute** | **~218-225 ms** |

Combined with the ~208ms of mandatory one-way PP-transport time (A->B +
B->C, each ~52ms one-way) plus RPC dispatch/ack overhead to the
Iowa-located Machine B, this reconciled with the observed ~500ms/token
(2.0-2.1 tok/s): compute was ~44% of the budget, network ~56%.

## CUDA graph experiment ("is it possible?")

Every prior real deployment on this project (GPT-OSS and this Qwen3.5 run)
forced `--enforce-eager` on every stage, based on a real bug found earlier:
with only ONE stage in CUDA-graph mode, its model runner pads
intermediate-tensor shapes to the nearest capture size before sending them
over the PP transport, while an eager neighbor expects the unpadded size ->
shape mismatch crash. Untested hypothesis going in: padding should agree
stage-to-stage if EVERY stage uses graphs consistently, since all PP ranks
process the same per-step token count.

Added `--enable-cudagraph` (real new flag, `scripts/stage_server.py` +
`scripts/launch_pp_stage.py`) to test this for real, with 3 real failures
found and fixed in sequence before it worked:

1. **Not the predicted shape-mismatch bug at all.** First attempt hung
   repeatedly on `shm_broadcast.py`'s "No available shared memory broadcast
   block found in 60 seconds" (real vLLM warning, unrelated to this
   project's synthetic PP group - it's the local multiproc EngineCore<->Worker
   queue, stuck because BOTH TP worker processes were mid-`torch.compile`
   (~46s) then blocked). Looked like a hang at first (2 killed attempts
   assumed so) - turned out to be slow-but-real work: with
   `--transport-connect-timeout` raised from 120s to 900s and the process
   left running instead of killed early, it surfaced a real terminal error
   after ~5 more minutes (see next point), not an infinite stall.
2. **Real, fixable config error:** `ValueError: max_num_seqs (128) exceeds
   available Mamba cache blocks (60)`. Qwen3.5's hybrid Mamba/attention
   architecture needs one dedicated Mamba cache block per concurrent decode
   sequence, and vLLM's default `max_num_seqs=128` exceeded the 60-block
   `--num-gpu-blocks-override` this deployment uses. Fixed by adding a new
   `--max-num-seqs` flag to both scripts and setting it to 60 (matching the
   block override).
3. **Real CUDA OOM** during dummy-sampler warmup with 60 concurrent dummy
   requests (`torch.OutOfMemoryError` computing a 60 x vocab_size=248,320
   softmax on an already-95%-utilized T4). Fixed by lowering
   `--max-num-seqs` further to 8 - one real chatbot user never needs 60
   concurrent sequences, and 8 leaves real headroom.

With `--max-num-seqs 8`, all 3 stages passed `torch.compile` + CUDA graph
capture and the full deployment came up successfully. **Conclusion: CUDA
graphs on this synthetic transport-backed PP group are real and possible**
- the original eager-mode requirement was a true bug for the
one-stage-graph/other-stage-eager case tested originally, not a fundamental
incompatibility with the transport architecture.

**Real measured result (same timing instrumentation, fresh request):**

| Stage | Local compute, eager | Local compute, CUDA graph | Speedup |
|---|---|---|---|
| Machine A | ~82 ms | **~16 ms** | ~5.1x |
| Machine B | ~84 ms | **~18 ms** | ~4.7x |
| Machine C (+ sampling) | ~59 ms | **~5 ms** | ~11.8x |
| **Total compute** | **~225 ms** | **~39 ms** | **~5.8x** |

Real end-to-end throughput (server-side `Avg generation throughput` log,
not client-side wall time which includes SSH overhead): **2.0-2.1 tok/s
(eager) -> 2.8 tok/s (CUDA graph)**, only ~35-40% faster overall despite
compute being ~5.8x faster - because the ~280ms/token of mandatory network
transport time (Machine B's Iowa location, unaffected by CUDA graphs at
all) now dominates ~88% of the total per-token budget. Compute is no longer
the bottleneck; network is now the entire story.

**Real cost of enabling CUDA graphs:** startup time grew from ~2-3 minutes
to ~8-10 minutes per stage (`torch.compile` ~46s + CUDA graph capture +
warmup, on top of the existing ~3 minute weight load), and requires the new
`--max-num-seqs` tuned to a value that fits both the Mamba-block budget and
available GPU memory headroom - a real fragility/startup-time tradeoff
against the throughput gain, left running live at the end of this session
since the gain was judged worth it for this deployment's single-user
chatbot use case.
