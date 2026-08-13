# Pipeline Decode-Degradation Bug: Root Cause, Fix, and Validation

**Status: FIXED and validated.** The custom transport-backed multi-machine
pipeline (`vllm/transport/`) now produces correct, coherent output across
decode steps. This was previously broken: any prompt longer than a couple
of tokens degraded into garbage (`"!!!!!!!"`, repeated tokens, or
character-level noise) starting from the second generated token onward.

This document records what was actually wrong, how it was found, and the
test that proves the pipeline itself (independent of any specific model)
is now correct.

## TL;DR

Two bugs were found and fixed in this session. Only the second one was the
actual cause of the garbage output; the first was a real but separate latent
bug fixed as defense-in-depth.

| # | File | Bug | Status |
|---|------|-----|--------|
| 1 | `vllm/transport/rpc_executor.py` | RPC responses from overlapping pipelined steps could get crossed on the same UDP link (no request/response correlation) | Fixed, but **not** the cause of the observed symptom |
| 2 | `vllm/v1/core/sched/scheduler.py` | `Scheduler.use_pp` was `False` on the driver, so decode-step `scheduler_output`s never carried the actually-sampled token id to non-driver stages | **This was the root cause** |

## Bug 2 (the real fix): `Scheduler.use_pp`

`vllm/v1/core/sched/scheduler.py`, `Scheduler.__init__`:

```python
self.use_pp = self.parallel_config.pipeline_parallel_size > 1
```

`use_pp` gates whether `_make_cached_request_data` populates
`CachedRequestData.new_token_ids` — the **only** mechanism by which a
non-last pipeline-parallel rank (which never runs sampling itself) learns
which token was actually sampled for a decode step. Without it, that stage
has no way to know what token to feed into its embedding lookup for the
next step.

This project's custom transport-backed PP (see
`scripts/launch_pp_stage.py`) deliberately runs every machine with a
**local** `--pipeline-parallel-size 1` — the real cross-machine PP group is
installed synthetically post-construction by `TransportPPWorker`, precisely
to avoid needing real `torch.distributed` rendezvous across NAT'd hosts.
That's the correct design for `is_first_rank`/`is_last_rank` and layer
partitioning (which read the *live* `_PP` group), but `Scheduler.use_pp`
reads the *config* value instead — which is always `1` in this deployment,
so `use_pp` was always `False` on the driver.

Effect: prefill worked fine (it carries `prompt_token_ids` directly, not
through `new_token_ids`). Every decode step after that forwarded an empty
`new_token_ids` to the non-driver stages, so the first stage's embedding
lookup ran on the wrong (or no) input token — corruption starting at the
second generated token, compounding every step after.

### The fix

```python
self.use_pp = self.parallel_config.pipeline_parallel_size > 1 or (
    int(os.environ.get("VLLM_TRANSPORT_PP_WORLD_SIZE", "1")) > 1
)
```

`VLLM_TRANSPORT_PP_WORLD_SIZE` is already set correctly (to the *real*
cross-machine stage count) by both `launch_pp_stage.py` and
`stage_server.py` — this reuses that existing signal rather than adding new
plumbing.

## Bug 1 (real, but not the cause): RPC crossed responses

`vllm/transport/rpc_executor.py`'s `TransportExecutor._dispatch_remote`
runs on a shared `ThreadPoolExecutor`. vLLM's own PP batch-queue pipelining
(`EngineCore.step_with_batch_queue`) can have multiple decode steps
in flight before earlier ones resolve, which meant two threads could,
in principle, call `send()`/`recv()` on the *same* `UDPTransport` link for
different steps concurrently. `UDPTransport.recv()` has no
request/response correlation id — just one shared FIFO queue — so a
`recv()` blocked on step N's reply could receive step N+1's instead.

Fixed by adding a `threading.Lock` per remote-stage link (serializes calls
to the *same* link only; different links still run concurrently), plus a
separate thread pool for the `_combine` step to avoid a related
thread-pool self-starvation/deadlock risk.

**This was verified NOT to be the cause of the garbage output**: the exact
same byte-for-byte garbage was produced before and after this fix. It's a
real correctness bug (worth having fixed) but was not being triggered in
this project's single-request test traffic pattern.

## Validation: Qwen2.5-7B-Instruct, split 3 ways

To isolate "is this a bug in our pipeline code, or in GPT-OSS's specific
model/quantization?", the same pipeline was tested against a completely
different, unrelated model: Qwen2.5-7B-Instruct — dense (no MoE),
unquantized (no GPTQ), different architecture family entirely.

**Single-machine baseline** (ground truth):
```
"What is 2+2?" -> "2 + 2 equals 4."
```

**3-way split, before the fix** — same class of corruption GPT-OSS showed:
```
"What is 2+2?" -> "2[]([]( + The ( The  !\n\n!\n\n!\n\n 22 22222 !\n\n!!\n\n!\n\n
The2222222 The The22222 The!!2222!!\n\n222 The2!!2222!2!"
```

**3-way split, after the fix** — byte-identical to the single-machine baseline:
```
"What is 2+2?" -> "2 + 2 equals 4."
```

Additional checks after the fix, same 3-way split deployment:
- Longer generation (82 tokens, factual paragraph about the Eiffel Tower):
  fully coherent, accurate.
- Non-English input (`"Apa ibu kota Indonesia?"`): fully coherent, correct
  Indonesian response.

This is strong evidence the pipeline itself — RPC forwarding, PP activation
tensor exchange, KV-cache block-table handling across 3 machines — is
correct, independent of model architecture or quantization.

### Reproducing the Qwen validation

Shard extraction (from a full Qwen2.5-7B-Instruct checkpoint):
```bash
python3 humming_fix/single_layer_probe/extract_stage_checkpoint.py \
  --src /path/to/qwen2.5-7b-instruct --start 0 --end 10 --out /data/qwen-stage0 --include-globals
python3 humming_fix/single_layer_probe/extract_stage_checkpoint.py \
  --src /path/to/qwen2.5-7b-instruct --start 10 --end 19 --out /data/qwen-stage1
python3 humming_fix/single_layer_probe/extract_stage_checkpoint.py \
  --src /path/to/qwen2.5-7b-instruct --start 19 --end 28 --out /data/qwen-stage2 --include-globals
```
(`10,9,9` split, not the naive `10,9,9` — matches vLLM's own uneven
28-layer/3-stage auto-partition; pass `VLLM_PP_LAYER_PARTITION=10,9,9` at
launch to force it explicitly rather than relying on auto-partitioning.)

Machine A (pp_rank 0):
```bash
VLLM_USE_V2_MODEL_RUNNER=0 VLLM_PP_LAYER_PARTITION=10,9,9 python3 scripts/stage_server.py \
  --model /data/qwen-stage0 --tensor-parallel-size 2 \
  --pp-rank 0 --pp-world-size 3 --self-name MachineA --next-name MachineB \
  --driver-name MachineC --transport udp --signaling-url $SIGNALING_URL \
  --dtype float16 --max-model-len 4096 --num-gpu-blocks-override 2048
```

Machine B (pp_rank 1):
```bash
VLLM_USE_V2_MODEL_RUNNER=0 VLLM_PP_LAYER_PARTITION=10,9,9 python3 scripts/stage_server.py \
  --model /data/qwen-stage1 --tensor-parallel-size 2 \
  --pp-rank 1 --pp-world-size 3 --self-name MachineB --prev-name MachineA --next-name MachineC \
  --driver-name MachineC --transport udp --signaling-url $SIGNALING_URL \
  --dtype float16 --max-model-len 4096 --num-gpu-blocks-override 2048
```

Machine C (pp_rank 2, driver + API server):
```bash
VLLM_USE_V2_MODEL_RUNNER=0 VLLM_PP_LAYER_PARTITION=10,9,9 python3 scripts/launch_pp_stage.py --serve \
  --model /data/qwen-stage2 --tensor-parallel-size 2 \
  --pp-rank 2 --pp-world-size 3 --self-name MachineC --prev-name MachineB \
  --remote-stage-names MachineA,MachineB --remote-stage-hosts 127.0.0.1,127.0.0.1 \
  --transport udp --signaling-url $SIGNALING_URL \
  --dtype float16 --max-model-len 4096 --num-gpu-blocks-override 2048 --port 8080
```

Launch all three together (not staggered by more than a couple minutes) —
the UDP hole-punch has a connect-timeout window per machine, and one
machine sitting alone waiting too long for the others can time out.

`VLLM_USE_V2_MODEL_RUNNER=0` is required for `Qwen2ForCausalLM` on this
setup: vLLM's V2 model runner caches `is_last_pp_rank` at construction
time, which is incompatible with `TransportPPWorker` installing the real
PP group *after* base `init_device()`. GPT-OSS never hit this because its
architecture already defaults to the V1 runner.

## GPT-OSS-120B: same fix applies, plus a separate, unrelated finding

The same `scheduler.py` fix resolves the identical decode-degradation
symptom on GPT-OSS-120B (GPTQ int4, MoE, 3 machines x 2xT4, TP=2/PP=3).
Short, simple prompts now produce clean, correct, naturally-terminated
output (`finish_reason: "stop"`).

On harder/longer reasoning prompts, GPT-OSS-120B sometimes still produces
rambling or occasional garbled tokens. This was investigated separately and
is **not a pipeline bug**:

- A residual-stream clamp in `vllm/model_executor/models/gpt_oss.py`
  (`_clamp_residual_fp16`, added in an earlier session) prevents a real
  fp16 overflow-to-NaN failure mode. Removing it (`VLLM_DISABLE_RESIDUAL_CLAMP=1`)
  reproduces the original catastrophic collapse (confirmed via NaN tracing:
  `compute_logits input: ... isnan=True`) — proving the clamp is load-bearing.
- With the clamp active, `compute_logits` shows **zero** NaN/inf across
  hundreds of calls, even on prompts that still produce rambling/garbled
  output — proving the residual clamp is *not* the cause of that separate,
  milder issue.
- The GPTQ checkpoint in use
  (`positron-ai/openai_gpt-oss-120b-ingest-best-gptq`) is explicitly
  documented (model card) as optimized for Positron's FPGA serving path,
  "runtime fidelity and FPGA deployability... prioritized over
  general-purpose GPU portability," with a measured 89.61% top-1 token
  agreement vs. the bf16 reference model. This is the most likely source of
  the remaining quality gap — a checkpoint-quality/calibration limitation,
  not a numerics or pipeline bug.
- `--dtype float32` is not an option: vLLM rejects it outright for GPTQ
  (`auto_gptq` supports only `float16`/`bfloat16`), and `bfloat16` is
  unavailable on T4 (compute capability 7.5, needs >= 8.0). fp16 (with the
  clamp) is the only valid dtype for this checkpoint on this hardware.
- `repetition_penalty` (sampling-time) measurably reduces the rambling on
  hard prompts; `temperature`/`repetition_penalty` cannot fix genuine NaN
  corruption (they operate on already-computed logits), which is why they
  were never expected to substitute for the clamp — confirmed empirically
  by disabling the clamp and testing them together (still collapses to
  garbage/NaN).

## Files changed this session

- `vllm/v1/core/sched/scheduler.py` — the actual fix (`use_pp`).
- `vllm/transport/rpc_executor.py` — RPC per-link locking + separate
  combine thread pool (real bug, not the cause of this symptom).
- `vllm/model_executor/models/gpt_oss.py` — added `VLLM_DEBUG_CLAMP_STATS`
  (clamp-hit frequency logging) and `VLLM_DISABLE_RESIDUAL_CLAMP` (A/B
  testing knob) to the pre-existing residual clamp from an earlier session;
  no behavior change to the default path.
