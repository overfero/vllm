# TODO — microbatching candidate follow-ups (2026-08-15)

Snapshot of open items from the microbatching (`step_with_batch_queue`)
work session. Candidate lives in `cluster/qwen35_122ba10b_3machine_pipelined.sh`
+ `vllm/transport/rpc_executor.py` (`--enable-pipelining`, `--enable-rpc-fusion`).
Baseline (`cluster/qwen35_122ba10b_3machine.sh`) is untouched/unaffected by
any of this - both stay independently runnable.

## Not yet deployed / verified live

- [ ] **Redeploy with `num_gpu_blocks_override=35`** (already changed in all
  3 cluster scripts, committed `4070531bb`) and confirm it actually starts
  and serves correctly - the fixes benchmarked so far (empty-batch skip +
  RPC fusion, +20-40% decode / +36-52% prefill / +43.8% concurrency n=8)
  were all measured with the OLD override=60 still in effect.

## Real, unresolved questions from this session

- [ ] **Root cause of the num_gpu_blocks auto-compute drop when pipelining
  is enabled.** Measured real, reproducible numbers:
  - Machine A: 182 (baseline) -> 182 (pipelined) - **no change**
  - Machine B: 171-172 (baseline) -> 127 (pipelined) - **-44/45 blocks**
  - Machine C: 82 (baseline) -> 37 (pipelined) - **-45 blocks**

  A is unaffected; B and C both lose ~45 blocks, and both are the machines
  with 2+ transport connections (B: 2 PP tensor links; C: 1 PP tensor link
  + 2 RPC links) vs A's 1 link. Not yet proven whether this is (a) the
  `batch_queue_size=3` -> `max_in_flight_tokens` KV-reservation mechanism
  (intentional, in `TransportExecutor._init_executor`'s
  `pipeline_parallel_size` mutation - only lives on the driver/C though,
  doesn't explain B), (b) the pipelined wire-protocol change itself
  (`VLLM_TRANSPORT_ENABLE_PIPELINING`, affects both B and C's
  `stage_server.py`/`rpc_executor.py` reader threads/transport buffering),
  or (c) something else. **To isolate**: redeploy with
  `--enable-pipelining` but WITHOUT `--batch-queue-size` (so
  `step_with_batch_queue` never actually activates) and recheck B/C's
  auto-computed block counts - if they're still ~127/~37, it's the wire
  protocol; if they're back near baseline, it's the batch-queue mutation.

- [ ] **Exact cause of Machine C's residual VRAM gap vs A/B beyond the
  pipelining-related drop above.** `embed_tokens`/`lm_head` are BOTH
  unconditionally constructed on every PP rank in `qwen3_next.py` (no
  `is_first_rank`/`is_last_rank` gate, unlike `norm` which IS gated) - so
  the earlier theory ("C carries an extra lm_head A/B don't") is not the
  full/correct explanation. Real weight-loading footprint measured:
  A=11.65 GiB, B=11.65 GiB (exactly equal), C=12.37 GiB (+0.72 GiB). Since
  construction is uniform across ranks, the +0.72 GiB on C must come from
  something checkpoint-content- or quantization-path-dependent, not from
  module construction itself - not yet root-caused with certainty.

- [ ] **Concurrency n=4 regression in v2 vs v1** (8.89 tok/s vs v1's 9.56,
  both still behind baseline's 12.19). n=1 and n=8 both improved a lot
  after the empty-batch-skip + RPC fusion fixes; n=4 didn't, and was only
  ever measured as a SINGLE sample (unlike n=8, which was re-verified with
  an 8-round/64-request average at 28.45 tok/s, +43.8% over baseline -
  the original single n=8 sample of 16.07 tok/s turned out to be a noisy
  outlier). n=4 needs the same multi-round averaging treatment before
  trusting the "still slower" conclusion.

- [ ] **No staggered-arrival baseline exists yet.** Only microbatch_v2's
  staggered-arrival number was measured (8 requests, 3s apart, 20.2 tok/s
  aggregate, per-request wall times decreasing from 46.1s down to 29.7s
  as later arrivals found less contention). This is the regime pipelining
  is specifically supposed to help with (new work schedulable without
  waiting for an in-flight step's full round trip) - need the SAME test
  against baseline (`cluster/qwen35_122ba10b_3machine.sh`) to know if
  microbatching actually wins here, since simultaneous-fire concurrency
  (n=1/4/8, all requests arriving at once) already gets fully captured by
  plain continuous batching regardless of pipelining.

## Real, understood, deliberately not pursued further (for now)

- [ ] **`async_scheduling=False`** - real speedup left on the table (hides
  CPU-side scheduling/bookkeeping behind GPU compute, a different overlap
  axis than pipelining's cross-machine one), but re-enabling currently
  hard-crashes (`AttributeError` on `pp.device_group` - the synthetic PP
  `GroupCoordinator` has no real `device_group`, since async scheduling's
  sampled-token broadcast bypasses this project's custom transport
  entirely and calls real `torch.distributed.broadcast` directly). Would
  need that specific broadcast call patched to route through the custom
  transport (same pattern as `sample_tokens`'s `pp_handler.broadcast`
  already does) before it's safe to try.

- [ ] **Asymmetric PP layer split** (e.g. 18/18/12 instead of 16/16/16) to
  give Machine C fewer real layers, freeing VRAM headroom to better match
  its structurally-heavier driver/API-server/RPC-dispatch role. This
  project's own history (git log, the 4-machine+MTP work) already proved
  asymmetric splits work correctly - just not applied to the current
  3-machine deployment. Would raise C's safe `num_gpu_blocks` ceiling
  without new hardware.

- [ ] **4th-machine "pure coordinator" design** - Machine D (excluded from
  this 3-machine deployment by design) could host ONLY the API
  server+Scheduler+RPC dispatch, no real model layers, making A/B/C all
  pure workers. Cleaner architecturally than the asymmetric-split option
  above but needs a machine that's currently idle/unused for this purpose.

- [ ] **`lm_head`/`embed_tokens` INT4 quantization** - currently BF16
  (confirmed via safetensors header inspection, not GPTQ-quantized despite
  earlier assumption otherwise). Could save ~0.36 GiB/GPU on the machines
  that hold them, at some real accuracy-risk cost to final-logit precision
  - not tested.

- [ ] **`--kv-cache-dtype fp8`** - not exposed in `stage_server.py`/
  `launch_pp_stage.py` yet (would need adding, same pattern as
  `--enable-expert-parallel`/`--enable-pipelining`). T4 (Turing, SM75) has
  no native FP8 tensor core - real benefit would be KV cache capacity
  (more concurrent long-context requests), not raw speed. Not tested.

- [ ] **MTP re-enablement** - this project's own history has a *proven*
  working config (12/12/12/12 uniform split + `--cpu-offload-gb 1` on
  every stage + CUDA graphs, real measured 6.4s vs ~20s eager, 92.9% draft
  acceptance) but it's for the old 4-machine layout, not applied to the
  current 3-machine/no-MTP deployment. `ngram`/`suffix` speculative
  decoding (zero extra VRAM, generic, lower acceptance rate than native
  MTP) is a lower-effort alternative never tried either.

## Already done and verified this session

- [x] Real `step_with_batch_queue` pipelining (`--enable-pipelining` +
  `--batch-queue-size`) - request-id-correlated RPC wire protocol replacing
  the old lock-serialized one, `TransportExecutor._local_wait_lock` fix for
  a real native `Bad address (src/fq.cpp:56)` libzmq crash. Committed
  `55c615e44`.
- [x] Empty-batch RPC skip + RPC fusion (`--enable-rpc-fusion`). Verified
  with real benchmarks: decode +20-40% vs baseline across all 8
  (context,output) combos tested, prefill +3-52% (bigger win at longer
  context), concurrency n=8 +43.8% (28.45 tok/s mean over 8 rounds,
  stdev 0.12). Committed `4070531bb`.
- [x] `num_gpu_blocks_override` 60 -> 35 reliability fix applied to all 3
  cluster scripts (real per-machine safe capacity measured: A=182, B=127,
  C=37 under pipelining - old 60 was above C's own safe threshold).
  Committed `4070531bb`. **Not yet redeployed/tested** (see top of file).
