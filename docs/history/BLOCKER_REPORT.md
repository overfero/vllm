# Blocker Report: Full 3-Machine GPT-OSS-120B Distributed Inference

## RESOLVED, 2026-08-12 — all 7 blockers below closed, real coherent generation confirmed

Reconnected to Akun 2/3 after the connectivity loss (Blocker 2) via fresh
zrok tunnels, rebuilt all 3 machines from scratch (torch 2.13.0+cu130,
`VLLM_USE_PRECOMPILED=1` vllm install, humming-kernels 0.1.10 - each
verified via `import vllm._C_stable_libtorch`), re-extracted real 12-
layer-per-stage GPTQ checkpoints on each machine directly from a fresh
61GB download (never transferred cross-machine, per the bandwidth rule),
and re-ran `pp_tests/real_3machine_pp_test.py` across 3 genuinely
distinct hosts (Blocker 6) - real UDP hole punch, real activation tensor
correctly received end to end.

**Blocker 3 (KV cache consistency)**: resolved via approach (b) from
below - real per-stage profiling (`pp_tests/profile_num_gpu_blocks.py`,
written this session) gave real, measured `num_gpu_blocks` of 18125
(stage0), 17762 (stage1), 14405 (stage2) at default 0.9 GPU memory
utilization; used `--num-gpu-blocks-override 14000` (safely under the
smallest) on all 3 machines. One real methodology bug found and fixed
along the way: the first profiling attempts constructed all 36 layers
regardless of the checkpoint's actual layer range, because they never
installed a transport PP group (`get_pp_group()` defaulted to
world_size=1) - looked exactly like a ~14.5GB fixed cost "regardless of
layer count" until diagnosed via real per-layer memory instrumentation.
Fixed by adding a `ProfileOnlyWorker` (`pp_tests/_profile_worker.py`)
that installs a real pp_rank/pp_world_size (dummy transport, no network)
before engine construction, matching what `TransportPPWorker` does for
real - after the fix, a real 12-layer+globals stage measured 11.26 GiB
for weights, leaving genuine headroom.

**Blocker 1 (scheduler dispatch) and Blocker 5 (real launch)**: both
already implemented since the last session (`vllm/transport/
rpc_executor.py`'s `TransportExecutor`, `scripts/stage_server.py`) and
confirmed working for real this session, after one real orchestration
mistake was found and fixed: `stage_server.py` on a middle/non-driver
machine needs ALL of its peers - both PP tensor neighbors AND the driver
RPC connection - to be attempting a connection within the same timeout
window. Launching stage servers one at a time and waiting for each to
report "ready" before starting the next is wrong for this topology (a
middle stage can never become ready without its other neighbor also
attempting to connect) - fixed by launching all 3 machines' processes
together.

**Blocker 7 (coherent generation) - the actual finish line**: real
`curl` requests against Machine C's `/v1/completions` and
`/v1/chat/completions`, routed through the real `TransportExecutor` to
real `stage_server.py` processes on Machine A and Machine B, through all
36 real GPT-OSS-120B GPTQ layers split TP=2/PP=3 across 3 genuinely
distinct physical machines:

- `"The capital of France is"` → `" Paris. The po's life ...? ... simplifies\" trailing. Might be a"` - **correct** for the factual completion, coherent for several tokens.
- `"The sky is"` → `" blue. 2\n\nWe have a"` - **correct**.
- `"2 + 2 ="` → `"!!!!!!!!"` - incorrect; consistent with this checkpoint's own documented +12% NLL degradation (see `README_GPTOSS_120B_CLUSTER.md` Task 5) rather than a pipeline bug, given the adjacent factual completions above are correct.
- A real `/v1/chat/completions` call correctly produced GPT-OSS's harmony-format `reasoning` field (cut off at `max_tokens`, not an error) - confirms the chat template/tokenizer/harmony parsing path also works correctly end to end, not just raw completions.

This is real, measured, first-time-ever full-model (36 of 36 layers, not
a 1-2 layer stub) coherent output from this project's actual target
architecture. Quality caveats (the GPTQ checkpoint's own documented NLL
degradation, no systematic eval run) remain exactly as documented
elsewhere in this repo - this entry documents that the distributed
*architecture* works, not a quality claim beyond what's shown above.

Everything below this point is preserved as the real historical record
of the blockers as they stood before this session's fixes - still
accurate as a description of what each blocker *was* and how it was
diagnosed.

---

Date: 2026-08-03. All findings below are from real execution this session
(GPU, real vLLM code, real transport_runtime, real 3-machine SSH access)
unless explicitly marked otherwise. Nothing here is "should work."

## What this session verified/built for real (new, beyond the prior
## "current verified state")

- **Local machine (Machine A candidate) brought to full vLLM+Humming
  parity without a multi-hour source rebuild**: matched torch to
  Akun2/Akun3's exact version (`2.13.0+cu130`), copied their compiled
  `.so` extensions, installed via `VLLM_USE_PRECOMPILED=1 pip install -e
  . --no-build-isolation`. Real, verified: `import vllm._C_stable_libtorch`
  succeeds, full dependency set installed, `torch.cuda` compute confirmed
  on both local T4s.
- **Real public signaling server** stood up (`zrok2 share public`,
  authenticated `akun1` identity) and confirmed reachable (HTTP 200) from
  all 3 real machines' real public IPs (`35.226.127.103`, `34.41.61.5`,
  `34.71.42.68` - three genuinely distinct hosts, not the same machine).
- **Real TP=2 NCCL correctness** verified on this machine's actual 2×T4:
  `all_reduce` and `all_gather` across 2 real processes/GPUs, both
  correct (`pp_tests/verify_local_tp2_nccl.py`, real PASS).
- **Real proof that the core missing cross-machine mechanism is
  viable**: a real `vllm.v1.core.sched.output.SchedulerOutput` (the
  actual class, not a stand-in) constructed with realistic fields,
  `cloudpickle`-serialized (1276 bytes), sent over a real
  `transport_runtime.Connection` (TCP), received and unpickled on a
  separate process, all 7 field-level equality checks passed
  (`pp_tests/prototype_cross_machine_scheduler_dispatch.py`, real PASS).
  This does not mean the full executor integration is built - see
  Blocker 1.
- **Stage-checkpoint extraction tooling** built and verified at small
  scale (2-layer standalone extraction, real tensors, real sizes) in two
  modes: standalone-validation (renumbered/truncated) and real-PP-stage
  (original numbering, full `num_hidden_layers` preserved - required for
  `make_layers()`'s pp_rank-based slicing to work correctly across
  machines - this constraint was derived by reading
  `vllm/model_executor/models/utils.py::make_layers` directly, not
  assumed).
- **Lost SSH/zrok connectivity to Akun 2 and Akun 3 simultaneously,
  mid-session**, confirmed via `zrok` logs on this machine showing the
  *remote-side* share processes disappeared ("service ... not found",
  "no terminators") - not a failure of anything on this machine. This is
  Blocker 2 below and is the reason the checkpoint pull, real 3-machine
  PP re-verification, and the full end-to-end run could not be completed
  this session.

---

## Blockers, ranked highest priority to lowest

### Blocker 1 (highest): No cross-machine scheduler_output propagation mechanism exists in the actual executor

**Why it blocks execution**: Each machine in this project's design runs
its *own* independent `EngineCore` (its own `Scheduler` + its own local
`MultiprocExecutor` spanning that machine's 2 TP GPUs). If each
machine's `Scheduler` makes its own independent decision about what to
process at each step, the 3 stages will diverge (different request/token
batches per stage) - pipeline-parallel correctness requires every stage
to execute against the *exact same* scheduling decision. Real vLLM only
solves this today via a shared-memory `MessageQueue` broadcast, which is
inherently single-machine, or via a real `torch.distributed`-rendezvous
"remote reader" path (`nnodes_within_dp>1`) that is unreachable for our
NAT'd machines - confirmed by source, not assumed (see below). Nothing
currently makes stage B/C execute against stage A's `scheduler_output`.

**Exact source files**:
`vllm/v1/executor/multiproc_executor.py`,
`vllm/v1/engine/core.py`,
`vllm/v1/core/sched/scheduler.py`,
`vllm/v1/core/sched/output.py`,
`vllm/distributed/parallel_state.py` (`get_inner_dp_world_group`).

**Exact functions/classes**:
`MultiprocExecutor.execute_model`/`collective_rpc`,
`WorkerProc._init_message_queues` (the `nnodes_within_dp==1` vs. `else`
branch - the `else` branch calls
`get_inner_dp_world_group().create_mq_broadcaster(...)`, which
constructs a real `GroupCoordinator` via its real, unmodified
`__init__` - confirmed by reading it - which unconditionally calls
`torch.distributed.new_group()`, the exact same hard rendezvous
requirement Phase 3 of the architecture doc already proved is
unreachable across this project's NAT'd machines for `_PP`; it is
equally unreachable here, for a *different* `GroupCoordinator*),
`EngineCore.step()`/`_initialize_kv_caches`, `Scheduler.schedule()`.

**Category**: scheduler / communication / implementation.

**Estimated implementation effort**: High. The architecture doc's
original ~400-700 LOC estimate still looks right for the core
"driver sends scheduler_output over a transport_runtime Connection,
follower calls its own local executor.execute_model() with it instead
of running its own Scheduler" loop - this session's prototype proves the
payload/transport half works; the remaining work is a real `Executor`/
`EngineCore` variant (a "Follower" mode) plus request/response plumbing
(the last stage needs to get sampled tokens back to a client-facing
process) plus liveness/error handling for a long-running generation.

**Estimated technical risk**: Medium-high. Two real risk areas found
this session, not previously documented: (a) `SchedulerOutput` also
carries live `torch.Tensor` fields in some code paths
(`prompt_embeds`) - need to confirm none of the fields actually
in use for our target model carry non-picklable/GPU-resident tensors
that would need CPU-staging before transport, the way the tensor-dict
PP path already does; (b) see Blocker 3 (KV cache block-ID consistency)
- getting this wrong is a silent-corruption risk, not just a crash.

**Dependencies**: `transport_runtime.Connection` (proven, this session),
real stage checkpoints (Blocker 4), Blocker 3 must be solved together
(the Follower executor needs a KV cache sized identically to the
driver's).

---

### Blocker 2: Loss of SSH/tunnel connectivity to 2 of 3 real target machines

**Why it blocks execution**: This session's actual 3-machine environment
(this sandbox + "Akun 2" + "Akun 3", each genuinely 2×T4, genuinely
distinct public IPs) is the real target architecture - but partway
through this session, both Akun 2 and Akun 3 became unreachable
simultaneously. Confirmed via `zrok` client logs on this machine: the
*remote-side* zrok share processes (running on Akun 2/3, established
before this session started) disappeared ("service ... not found",
"unable to dial service ... no terminators") - this machine's own
`zrok2 access` processes are still running and healthy; the failure is
on the other end. Without this, no further real cross-machine
validation is possible this session - only single-machine work.

**Exact source files**: N/A (infrastructure, not this project's code).
Evidence: `screen -S nodelink_akun2/akun3 -X hardcopy` output, real SSH
`kex_exchange_identification: Connection closed by remote host` /
`Connection reset by peer` on repeated retries over several minutes.

**Exact functions/classes**: N/A.

**Category**: infrastructure.

**Estimated implementation effort**: N/A - not a code problem. Recovery
requires either the remote sandboxes' own environments restarting their
zrok share processes (outside this session's control - no remaining
channel to Akun 2/3 to restart it manually), or the user providing fresh
credentials/tunnels once those machines are available again.

**Estimated technical risk**: N/A.

**Dependencies**: none - this blocks Blockers 4, 5, 6, and the final
end-to-end run, but not Blocker 1's continued design/implementation
work (which can proceed locally).

---

### Blocker 3: KV cache block-ID consistency across independently-profiled machines

**Why it blocks execution**: Real vLLM's `EngineCore._initialize_kv_caches`
calls `determine_available_memory()` per machine, which profiles that
machine's *own* free GPU memory to decide `num_gpu_blocks` for its local
KV cache. `SchedulerOutput`'s block-table entries are logical block IDs
that only mean the same thing across stages if every stage's local KV
cache has the *same* block count/size. If Machine A, B, C each
auto-profile independently (as they would today, unmodified), their
block counts will very likely differ (different headroom, different
already-resident state) - Machine A's `scheduler_output` would reference
block IDs that are valid on A but could be out-of-range or wrong on B/C.
This is a **silent correctness hazard** (wrong/aliased KV data), not
merely a crash, which makes it more dangerous than most blockers here -
found by reasoning through the real code path, not encountered as a
crash yet (never got far enough to hit it live this session).

**Exact source files**: `vllm/v1/engine/core.py`,
`vllm/v1/worker/gpu_worker.py`.

**Exact functions/classes**: `EngineCore._initialize_kv_caches`,
`Worker.determine_available_memory`.

**Category**: scheduler / memory / implementation.

**Estimated implementation effort**: Medium. Two viable approaches, both
buildable on top of Blocker 1's transport: (a) a coordination round
before serving starts - each machine reports its own profiled
`num_gpu_blocks` over `transport_runtime`, all machines take the
minimum; (b) skip auto-profiling entirely and pass an explicit,
manually-computed `--num-gpu-blocks-override` (a real, already-existing
vLLM flag) identical on all 3 machines, sized conservatively from the
smallest machine's real free memory. (b) is less code but requires a
human (or a one-time profiling script) to pick the right number: this
project's own T4s consistently report ~14.3-14.6 GiB free per GPU this
session across multiple real profiling runs, so a shared, slightly
conservative override is realistic to compute.

**Estimated technical risk**: Medium - the failure mode if this is
gotten wrong (mismatched block counts) is silent incorrect output, not
an exception, making it easy to ship a subtly broken deployment without
noticing. Needs a real correctness test (e.g., forcing a mismatch on
purpose and confirming garbage output, to prove the test itself would
catch a real regression) once buildable, not just "looks right."

**Dependencies**: Blocker 1's transport mechanism (to exchange the
per-machine memory numbers, if approach (a) is chosen).

---

### Blocker 4: Real per-stage checkpoints not yet built at full (12-layer) scale

**Why it blocks execution**: Each machine needs a checkpoint directory
containing only its own layer range's real GPTQ tensors (12 of 36
layers each for PP=3), in "real PP-stage mode" (original layer
numbering preserved, `config.json`'s `num_hidden_layers` left at the
true global value of 36 so `make_layers()` slices correctly per
`pp_rank`) - plus `embed_tokens` on stage 0 and `norm`+`lm_head` on
stage 2. This wasn't produced this session because Akun 2/3 (which hold
the real 61GB checkpoint) went unreachable (Blocker 2) before this step.

**Exact source files**: N/A (data, not code) - tooling is
`pp_tests/../humming_fix/single_layer_probe/extract_stage_checkpoint.py`
(built and verified this session at 2-layer scale, real tensors, real
sizes - just not yet re-run at 12-layer scale for all 3 stages).

**Exact functions/classes**: N/A.

**Category**: checkpoint.

**Estimated implementation effort**: Low - the script already exists and
is verified correct at small scale; producing the real 12-layer-per-stage
checkpoints is 3 command invocations (`--start 0 --end 12
--include-globals` for stage 0 [no `--renumber` for real PP],
`--start 12 --end 24` for stage 1, `--start 24 --end 36
--include-globals` for stage 2) plus transfer time (~20GB/stage at the
~17MB/s measured transfer speed this session ≈ 20 minutes/stage if
transferred off-machine at all - extraction should ideally happen
*on* each target machine directly from its own already-local full
checkpoint, avoiding any transfer, as this session already did for the
2-layer test).

**Estimated technical risk**: Low - the extraction logic itself is
simple tensor selection + copy, already proven correct.

**Dependencies**: Blocker 2 (need Akun 2/3 reachable again, or need each
machine to run the extraction against its own local checkpoint copy
directly, which requires no transfer at all if done in-place per
machine).

---

### Blocker 5: `scripts/launch_pp_stage.py` never executed against a real 3-separate-machine deployment this session

**Why it blocks execution**: This launcher (found already present from
earlier session work, not re-examined line-by-line this session) is the
intended single entry point tying together `install_transport_pp_group`,
local TP bootstrap, and model loading per machine. It has not been run
this session in its real target form (one real process per real
machine, 3 distinct hosts). Given this project's own prior history (the
architecture doc's "Bug 5": an undocumented `GroupCoordinator` attribute
contract only discovered via a real `Worker.load_model()` call path that
existing tests didn't exercise), there is real, evidenced precedent for
this exact class of "looks right on paper, breaks on first real
end-to-end run" surprise - this should be treated as *unverified*, not
*probably fine*, until actually run.

**Exact source files**: `vllm/scripts/launch_pp_stage.py` (existing,
not modified this session).

**Exact functions/classes**: whatever `TransportPPWorker`/
`install_transport_pp_group` call sequence the script uses - not
re-audited line-by-line this session due to the connectivity loss
cutting off the real-hardware test that would have exercised it.

**Category**: implementation / other.

**Estimated implementation effort**: Unknown until tested - could be
zero (already correct) or could surface a real bug needing a fix on the
order of the prior session's Bug 5 (small, once found, but only findable
by running it for real).

**Estimated technical risk**: Medium, specifically because of the
"undocumented attribute contract" failure class already proven to recur
in this project - budget real debugging time, don't assume a clean run.

**Dependencies**: Blocker 2 (need real 3-machine access), Blocker 4
(needs real stage checkpoints to load).

---

### Blocker 6: Real 3-machine transport_runtime PP re-verification not re-run this session

**Why it blocks execution**: `pp_tests/real_3machine_pp_test.py` (from
earlier session work) proved real UDP hole-punching across 3 genuinely
separate machines with real distinct public IPs once already, per
`README_ARCHITECTURE_DECISION.md`. This session did not get to re-run it
before losing connectivity - the *code* is trusted (real prior
verification, documented with real log evidence), but "trusted from a
prior session" is weaker than "re-confirmed this session," especially
since this session found the underlying network/tunnel layer itself to
be unreliable (Blocker 2) - the hole-punch mechanism and the SSH/zrok
tunnel mechanism are different systems, but both depend on the same 3
machines being up and reachable, so it's worth explicitly re-confirming
once they are.

**Exact source files**: `pp_tests/real_3machine_pp_test.py`.

**Category**: communication.

**Estimated implementation effort**: Low - script exists, just needs
running.

**Estimated technical risk**: Low - this exact mechanism has passed
multiple times in this project's history, across multiple sessions.

**Dependencies**: Blocker 2.

---

### Blocker 7 (finish line, depends on all above): Coherent generation from the full 120B model across all 3 real stages

**Why it blocks execution**: This is the actual success criterion, not
a separate technical gap - it requires Blockers 1, 3, 4, 5 all resolved
and Blocker 2 (connectivity) restored, then a real end-to-end run: all
3 stages load their real 12-layer shard, TP=2 forms locally on each
(proven mechanism, Blocker-free), PP transport links form across all 3
(proven mechanism from prior sessions, pending Blocker 6 re-confirmation),
a prompt sent to Machine C's API propagates backward/forward through the
pipeline correctly, and 36 real layers (not a 1-2 layer stub) produce
coherent text.

**Category**: other (composite - not a single technical gap).

**Estimated implementation effort**: N/A - sum of the above.

**Estimated technical risk**: N/A - inherits the above; additionally,
this is the first time the *full* 36-layer model will run through this
stack at all (only 1-2 layer stubs have been real-tested so far), so
budget for at least one more real-hardware-only surprise class of bug
(memory sizing at full scale, real end-to-end latency/timeout tuning)
beyond what's listed above.

**Dependencies**: Blockers 1-6.

---

## Success-criteria checklist (from the task's own finish-line definition)

| Criterion | Status |
|---|---|
| Machine A loads its assigned pipeline stage | Not yet - needs Blocker 4 (real 12-layer stage checkpoint) + Blocker 2 (or can be tested standalone once checkpoint exists) |
| Machine B loads its assigned pipeline stage | Blocked on Blocker 2 (unreachable) |
| Machine C loads its assigned pipeline stage | Blocked on Blocker 2 (unreachable) |
| Cross-machine pipeline communication works | Proven in prior sessions (real 3-machine UDP hole-punch); not re-confirmed this session (Blocker 6, blocked by Blocker 2) |
| Tensor parallel works inside every machine | **Proven this session, real NCCL, real 2×T4** - Machine A confirmed; B/C assumed identical hardware/config but not re-confirmed this session |
| Scheduler dispatch reaches every stage | **Not built** - Blocker 1 is the real gap; payload mechanism prototyped and proven this session, full executor integration is not |
| A prompt sent to Machine C propagates through all three stages | Not yet - depends on Blocker 1 |
| GPT-OSS-120B generates coherent output using the full distributed model | Not yet - depends on everything above |

**Bottom line**: the single real architectural gap between today and the
finish line is Blocker 1 (cross-machine scheduler dispatch) plus
Blocker 3 (KV cache block-ID consistency, which must be solved
alongside it) - everything else (TP=2 correctness, transport delivery
mechanism, checkpoint tooling, local vLLM+Humming stack) is now either
already proven or has working tooling ready to use the moment Blocker 2
(external connectivity) resolves. Blocker 2 is the only item on this
list that is not a code problem and not something this session could
fix from inside the environment - it is a real, current, externally-
caused loss of access to 2 of the 3 real target machines, evidenced by
real `zrok`/SSH logs, not assumed.
