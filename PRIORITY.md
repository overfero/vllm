# PRIORITY — resume here (2026-08-16)

See also `TODO.md` for older, lower-urgency open items from the previous
session (num_gpu_blocks_override validation, block-count-drop
investigation, etc.) — this file is the current active thread.

## Current blocker

**Akun4 (Machine C, the driver) disconnected mid-benchmark** -
`Connection reset by peer` / `kex_exchange_identification` on every SSH
retry. Same pattern Akun3 hit earlier this session. The whole cluster is
down. User reported the Kaggle session itself showed a Docker daemon
connection error suggesting possible disk exhaustion - checked disk usage
directly on Machine A and Akun3 (both reachable at the time): both
healthy, ~997GB free, `/tmp` negligible, torch compile cache only
~430MB. **Nothing in our own deploy process explains this** - most likely
a Kaggle platform/host-level issue, not a bug to chase in our scripts.

**To resume:** check/restart Akun4's Kaggle session, get a fresh SSH
port+password, update `.env`'s `MACHINE_C_PORT`/`MACHINE_C_PASSWORD`,
redeploy with `bash cluster/qwen35_122ba10b_3machine_pipelined.sh`.

## CLOSED: GIL switchinterval fix - investigated, tried, reverted

Full arc, so this doesn't get re-litigated from scratch later:

1. Found real RPC round trips (20-50ms) were far slower than raw network
   (`pp_tests/real_ping_pong.py`: 1.21ms RTT between co-located machines,
   confirmed with both Akun3 and Akun4, 0% loss). Ruled out payload size
   with a size-matched ping-pong test (2000 bytes, same ~1.3ms result).
2. Hypothesized GIL contention (Python's default `sys.setswitchinterval()`
   is 5ms; many threads per process - reader threads, per-link asyncio
   loops, `_combine_pool` workers - competing for the GIL). A synthetic
   benchmark reproducing the same shape supported this (mean 40ms->7.2ms
   improvement at a shorter interval).
3. Implemented `sys.setswitchinterval(0.0005)` in
   `TransportExecutor._init_executor()` and `stage_server.py`'s `main()`,
   deployed live, benchmarked N=1/2/4 against the pre-fix baseline.
4. **Result: no improvement, real regression at N=4 (-27.8%, 49.59->35.8
   tok/s aggregate)**, N=1/N=2 also slightly down. Directly contradicted
   the synthetic benchmark's prediction.
5. **Root cause of why the fix didn't help, found via direct before/after
   comparison of the SAME metric on the SAME machines**: hop send time
   was already ~1.3-1.4ms median BEFORE the fix, ~1.5-1.65ms after -
   essentially unchanged. The fix never had anything real to improve;
   the RPC/transport layer was never the bottleneck it was built to fix.
6. **Reverted** (commit after `3972245f5`) - removed
   `sys.setswitchinterval(0.0005)` from both files, `sys` import cleaned
   up from `rpc_executor.py` (was otherwise unused there). Not worth
   keeping unproven custom code with a demonstrated regression risk.

## ACTUAL finding: per-stage compute time is the real latency driver, not transport

Broke down a single (N=1) request's per-token latency using the existing
`[EXECUTE_MODEL_TIMING]` (total duration of `TransportPPWorker.execute_model` -
recv-from-prev + local compute + send-to-next) and `[TRANSPORT_TIMING]`
(real send/recv duration on the wire, logged unconditionally by
`udp_transport.py`, no debug flag needed) - both already existed in the
code before this session, just hadn't been read together before.

| Stage | Total/step | Real hop (send, measured) |
|---|---|---|
| A (first) | 17.0ms | ~2.9ms |
| B (middle) | 27.5ms | ~2.9ms |
| C (last, driver) | ~39.5ms (median of non-filler steps) | - (no next stage) |

**Real transfer time is small (~2.9ms/hop, ~6ms total across 2 hops) -
this is NOT the bottleneck.** Per-stage compute is what dominates,
especially Machine C (~39.5ms) - the last stage carries the full
`lm_head` (projects to full vocab, a big matmul), sampling logic, AND
driver/API-server/scheduler overhead all in the same process. This is
consistent with earlier (unrelated) findings this session about C's
structurally heavier role.

Caveat: summing all 3 stages (17.0+27.5+39.5=84ms) doesn't cleanly match
the observed ~60ms/token at N=1 - each stage's own `recv` duration is
inclusive of upstream wait time (not pure network), so there's some
double-counting risk in a naive sum, and/or genuine partial overlap even
at N=1. Not fully reconciled to the ms - the qualitative conclusion (hop
is small, compute dominates, C is heaviest) is solid; the exact
decomposition isn't.

**Where to look next if resuming this thread**: reducing Machine C's own
compute load (asymmetric split giving C fewer real layers - proven
possible, see `docs/DEPLOYMENT.md`'s "Asymmetric PP splits" - or offloading
the API server/scheduler to a dedicated non-compute machine, see this
project's own older "4th-machine pure coordinator" idea in the previous
session's TODO.md) is the evidence-backed lever for individual
(single-request) latency - NOT the transport layer, which is already
about as fast as raw network allows.

## Data collected this session (for reference, not yet fully used)

Pre-GIL-fix baseline (full N=1,2,4,6,8 sweep, compute+hop timing
included) backed up at
`/tmp/claude-0/-kaggle-working/0e01be65-4d72-46f1-86eb-799cd4663b95/scratchpad/baseline_before_gilfix/`
(20 files). Real numbers:

| N | Aggregate tok/s | Per-request tok/s |
|---|---|---|
| 1 | 16.81 | 16.81 |
| 2 | 35.01 | 17.53 |
| 4 | 49.59 | 12.42 |
| 6 | 60.94 | 10.17 |
| 8 | 78.25 | 9.82 |

Scratchpad note: this environment has reset/wiped scratchpad contents
multiple times this session, unprompted - don't assume anything under
`/tmp/claude-0/.../scratchpad/` survives indefinitely. Benchmark/sweep
scripts may need recreating if gone; straightforward to rebuild
(concurrent `urllib` requests to `/v1/completions`, log-line-diffing per
concurrency level) - ask if needed rather than guessing their old shape.

## How to reproduce the debug-timing deploy (for [EXECUTE_MODEL_TIMING] /
## [stage-exec-timing] / [rpc-pipelined-timing])

The committed `cluster/qwen35_122ba10b_3machine_pipelined.sh` does NOT
set `VLLM_TRANSPORT_DEBUG_TIMING=1` by default. To get that breakdown:
copy the script, inject `VLLM_TRANSPORT_DEBUG_TIMING=1` before each of
the 3 stage-launch commands (local `nohup` for Machine A, `export
VLLM_TRANSPORT_DEBUG_TIMING=1 &&` inside each remote SSH command string
for B/C), fix the `cd "$(dirname ...)/.."` line to a hardcoded
`cd /kaggle/working/vllm` if run from outside `cluster/`. Note:
`[TRANSPORT_TIMING]` (used for the compute-vs-hop breakdown above) is
logged unconditionally by `udp_transport.py` regardless of this flag -
only `[EXECUTE_MODEL_TIMING]`, `[stage-exec-timing]`, and
`[rpc-pipelined-timing]` need it.

## Known deploy-script gotchas hit again this session

- `pkill -f 'pattern'` run inside a single SSH command string can match
  the wrapping `bash -c "..."` process itself (its cmdline literally
  contains the pattern text) and kill the SSH session before later
  commands in the same chain run - run each `pkill` as its own separate
  SSH call.
- The deploy script's own `flock`-based lockfile can get "stuck held" by
  a legitimate long-running child (the signaling server, forked before
  the script's own `wait`) even after the main script process exits -
  safe fix is `rm -f` the lockfile (does not disturb the still-running
  signaling server, which holds the fd but never calls flock() itself).
- If the deploy script's own wrapper process doesn't fully exit after an
  error (stuck in `wait` on a detached SSH session), find and `kill -9`
  it specifically - don't touch the signaling server / zrok tunnel
  process it may have forked.
- SSH host keys change every time a Kaggle account's session restarts
  (fresh container = fresh host key) even if the port number is reused -
  `ssh-keygen -f "/root/.ssh/known_hosts" -R "[127.0.0.1]:<port>"` before
  retrying if you see "REMOTE HOST IDENTIFICATION HAS CHANGED".
