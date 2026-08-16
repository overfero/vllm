# PRIORITY — resume here (2026-08-16)

Blocked mid-optimization-pass. Everything needed to pick this back up
exactly where it left off is below. See also `TODO.md` for older,
lower-urgency open items from the previous session (num_gpu_blocks_override
validation, block-count-drop investigation, etc.) — this file is just the
current active thread.

## Current blocker

**Akun3 (Machine B in the pipeline) disconnected mid-benchmark** -
`Connection reset by peer` / `kex_exchange_identification` on every SSH
retry, consistent across multiple attempts, not a transient blip (Kaggle
free-tier session almost certainly ended/timed out). Machine B going away
during a live request made Machine C (the driver) crash and fully shut
down too - **the whole cluster is down right now**, not just one machine.

**To resume:**
1. Check/restart Akun3's Kaggle session, get a fresh SSH port+password.
2. Update `.env`'s `MACHINE_B_PORT`/`MACHINE_B_PASSWORD` (Akun3) - Akun4
   (`MACHINE_C_PORT`/`MACHINE_C_PASSWORD`) should still be fine, worth a
   quick reachability check first though.
3. Redeploy: `bash cluster/qwen35_122ba10b_3machine_pipelined.sh` (or the
   debug-timing-enabled scratchpad variant if profiling is still wanted -
   see "How to reproduce" below, it doesn't survive scratchpad resets so
   may need rebuilding).
4. Re-run the concurrency sweep (N=1,2,4,6,8) and compare against the
   backed-up pre-fix baseline (see "Data already collected" below).

## What's already done

- **Root-caused why real RPC round trips (20-50ms) were ~20-40x slower
  than raw network** (confirmed via `pp_tests/real_ping_pong.py`: 1.21ms
  RTT between Machine A and both Akun3 and Akun4, co-located Iowa
  datacenter, 0% loss). Ruled out payload size as the cause with a
  size-matched ping-pong test (2000 bytes, same result: ~1.3ms) - the
  actual RPC dispatch path was measuring 20-50ms for the same size
  payloads, so neither network nor serialization-over-wire explained the
  gap.
- **Identified GIL contention as the real cause**: Python's default
  `sys.setswitchinterval()` is 5ms; with multiple threads per process
  (reader threads, per-link asyncio event loop threads, `_combine_pool`
  workers, all contending against the main EngineCore scheduling loop),
  a thread with real work ready (e.g. a reader thread whose recv() queue
  just got a reply) can sit unscheduled for up to one switch interval. A
  synthetic benchmark reproducing the same shape (CPU-bound threads +
  queue-based responder/waiter) measured mean 40ms/max 549ms at the
  default 5ms interval vs mean 7.2ms/max 28ms at 0.5ms - closely matching
  the real numbers observed.
- **Fix implemented and committed**: `sys.setswitchinterval(0.0005)` added
  early in `TransportExecutor._init_executor()`
  (`vllm/transport/rpc_executor.py`) and `stage_server.py`'s `main()`.
  Commit `062a5d6c8`, pushed.
- **Partially verified live**: redeployed with the fix
  (`vllm-0.1.dev24+g062a5d6c8`), sanity check passed (correct output),
  N=1 benchmark matched pre-fix baseline exactly (16.81 tok/s - expected,
  N=1 has no concurrent threads to contend over, so no difference should
  show there). **N=2/4/6/8 comparison never completed** - Akun3 dropped
  during the N=1 step of the sweep script itself (the very first
  concurrent-load test), before any of the levels that would actually
  exercise the fix could run.

## Data already collected (for the eventual before/after comparison)

Pre-fix baseline (full N=1,2,4,6,8 sweep, compute+hop timing included) is
backed up at
`/tmp/claude-0/-kaggle-working/0e01be65-4d72-46f1-86eb-799cd4663b95/scratchpad/baseline_before_gilfix/`
- 20 files (`sweep_n*.jsonl` for throughput, `a/b/c_timing_n*.txt` for the
  raw `stage-exec-timing`/`rpc-pipelined-timing` log lines per level).
Real numbers already reported to the user mid-session:

| N | Aggregate tok/s (pre-fix) | Per-request tok/s (pre-fix) |
|---|---|---|
| 1 | 16.81 | 16.81 |
| 2 | 35.01 | 17.53 |
| 4 | 49.59 | 12.42 |
| 6 | 60.94 | 10.17 |
| 8 | 78.25 | 9.82 |

Post-fix: only N=1 confirmed (16.81 tok/s, matches). **N=2/4/6/8 not yet
re-measured with the fix** - this is the actual point of doing this pass,
still outstanding.

Scratchpad note: this environment has reset/wiped scratchpad contents at
least twice already this session (once mid-conversation, unprompted) -
don't assume anything under `/tmp/claude-0/.../scratchpad/` survives
indefinitely. `benchmark_6agents.py` and `full_profiling_sweep.sh` may
need to be recreated if gone; their content is reconstructable from this
conversation's history if needed, or ask - they're straightforward
(concurrent `urllib` requests to `/v1/completions`, log-line-diffing per
concurrency level).

## How to reproduce the debug-timing deploy

The committed `cluster/qwen35_122ba10b_3machine_pipelined.sh` does NOT
set `VLLM_TRANSPORT_DEBUG_TIMING=1` by default (adds log noise, not meant
for normal runs). To get the `[stage-exec-timing]` /
`[rpc-pipelined-timing]` breakdown again: copy the script, inject
`VLLM_TRANSPORT_DEBUG_TIMING=1` before each of the 3 stage-launch
commands (local `nohup` for Machine A, and `export
VLLM_TRANSPORT_DEBUG_TIMING=1 &&` inside each remote SSH command string
for B/C), fix the `cd "$(dirname ...)/.."` line to a hardcoded
`cd /kaggle/working/vllm` if run from outside `cluster/` (e.g. from
scratchpad).

## Known deploy-script gotchas hit again this session (already documented
## in commit messages / code comments, listed here just so they don't
## cost time twice)

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
