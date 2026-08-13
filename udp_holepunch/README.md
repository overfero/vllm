# Direct UDP peer-to-peer network benchmark

Two peers behind NAT, no public IP on either side, establish a **direct UDP
connection** using hole punching, coordinated by a tiny HTTP signaling server.
The signaling server only ever exchanges endpoint metadata over HTTP — it
never sees or relays a single byte of UDP traffic. Once the hole is punched,
`peer.py` runs an instrumented network benchmark suite over that direct link:
RTT/jitter/loss, sustained throughput, duplex contention, latency under load,
packet reordering/duplication, NAT keepalive survival, NAT rebinding
detection, CPU utilization, and a check for whether Python itself — not the
network — is the throughput ceiling.

## Files

- `signaling_server.py` — FastAPI app, two endpoints, in-memory state, no auth, no relay.
- `peer.py` — asyncio, one UDP socket for everything: STUN, hole punch, and every benchmark below.

## Install

On the coordinator host and on both peers:

```bash
pip install fastapi uvicorn requests
```

(`peer.py` implements its own minimal STUN client — no `pystun3`/STUN
library dependency, so that exactly one UDP socket is ever created, and
`resource` for CPU sampling is stdlib-only, POSIX.)

## How it works

1. Each peer registers `{peer_id, udp_port}` with the coordinator. The
   coordinator determines the peer's **public IP from the HTTP request
   itself** (`X-Forwarded-For` if present, else the TCP peer address) — it
   never trusts a client-supplied IP.
2. Once both peers are registered, the next `GET /peer/{id}` call from
   either side gets back the other peer's endpoint **plus a shared
   `start_at` UNIX timestamp**, computed once and frozen so both peers see
   the same value. Both peers wait until that instant before sending their
   first punch packet — this is what makes the punch simultaneous instead
   of racing on however fast each side happened to poll.
3. Both peers fire UDP packets at each other's public endpoint at
   `start_at`. Each side's outbound packet opens its own NAT's mapping at
   (as close as possible to) the same moment the other side's inbound
   packet arrives — that's the actual "hole punch."
4. From then on, `peer.py` runs the benchmark suite directly over that same
   socket: no further coordinator involvement, and every subsequent phase
   transition (each transfer, each streaming test) is synchronized between
   the two peers with an app-level `barrier()` exchanged over that same
   direct UDP channel — not by hoping both sides' clocks and local timing
   happen to line up.

## Running it

### 1. Coordinator (run once, anywhere reachable by both peers)

```bash
python3 signaling_server.py
# listens on 0.0.0.0:8000
```

Expose it through zrok so both peers can reach it over the internet:

```bash
zrok share public 8000
# -> prints a public https URL, e.g. https://abcd1234.share.zrok.io
```

Keep this running for the whole test. If you re-run it, both peers must
re-register (they do this automatically on startup).

### 2. Server A

```bash
python3 peer.py --id A --peer-id B \
    --signaling-url https://abcd1234.share.zrok.io \
    --mode stun
```

### 3. Server B

```bash
python3 peer.py --id B --peer-id A \
    --signaling-url https://abcd1234.share.zrok.io \
    --mode stun
```

Start both within a minute or so of each other — the coordinator gives a
5-second lead time once both have registered, so there's no need to launch
them in the exact same instant.

Run with `python3 -u peer.py ...` (unbuffered) or set `PYTHONUNBUFFERED=1`
if you're piping output to a file/log and want to see it live rather than
only on exit.

### Modes

- `--mode stun` (default): each peer runs a STUN Binding Request against
  `--stun-host`/`--stun-port` (default `stun.l.google.com:19302`) **on the
  same socket** it will use for everything else, to learn its real public
  `ip:port`, then registers that.
- `--mode preserve`: skips STUN entirely. Only `udp_port` (the local bound
  port) is registered; the coordinator supplies the public IP from the HTTP
  request. This assumes the NAT preserves the local port 1:1 as the public
  port. Use this if you've already confirmed (e.g. via the `stun` CLI
  against `stun.l.google.com:19302`) that both servers' NATs preserve ports
  — it saves a STUN round trip and works even if outbound access to STUN
  servers is blocked.

## Experiments

`--experiment` picks what runs after the hole punch and the idle-RTT
baseline (both always run first, on every experiment, so there's always an
idle-baseline to compare a loaded number against).

| `--experiment` | What it runs | Maps to |
|---|---|---|
| `default` (no flag needed) | Transfer benchmark, payload benchmark, one short uni-directional pair, one short duplex+loaded-latency run, NAT keepalive test | A practical single-run smoke test covering everything at once |
| `payload-sweep` | One-shot transfers at 256KB → 32MB | Experiment A |
| `long-streaming` | Uni-directional continuous streaming, 300s each direction (leader sends first) | Experiment B |
| `long-duplex` | Simultaneous bidirectional streaming, 300s | Experiment C |
| `loaded-latency` | Duplex stream + 100ms-interval pings, reports idle vs. loaded RTT | Experiment D |
| `multi-stream` | 1/2/4/8/16 concurrent duplex streams over the one socket, reports aggregate throughput scaling | Experiment E |
| `fast-path-compare` | The same duplex stream run twice — normal path, then a preallocated-buffer/raw-socket fast path — to isolate Python overhead | Experiment F |

Every mode accepts `--stream-duration <seconds>` to override its default
duration (20s for `default`'s streaming pieces, 300s for the two `long-*`
modes) — useful for a quick sanity check before committing to a 5-minute
run: `--experiment long-duplex --stream-duration 15`.

### Other flags

| Flag | Default | Purpose |
|---|---|---|
| `--port` | random 20000-60000 | local UDP port (also used for STUN) |
| `--quick` | off | shrinks sizes/durations/probe-counts for a fast smoke test (seconds instead of minutes); ignored by the two `long-*` experiments unless combined with `--stream-duration` |
| `--stun-host` / `--stun-port` | Google STUN | override the STUN server |
| `--pacing {max,mbps,pps}` | `max` | cap the streaming send rate to a fixed Mbps or packets/sec instead of sending as fast as possible |
| `--pacing-value` | — | the target rate for `--pacing mbps`/`pps` |
| `--fast-path` | off | use the preallocated-buffer/raw-socket send path for every streaming test in this run (not just `fast-path-compare`) |

For a first run, or to sanity-check the setup before committing to a long
benchmark, add `--quick` on both sides with `--experiment default`.

## Interpreting the output

### Setup and connectivity

- **Socket buffers** — what was requested vs. what the kernel actually
  granted for `SO_RCVBUF`/`SO_SNDBUF`. A grant far below the 4MB request
  usually means `net.core.rmem_max`/`wmem_max` are capped low on that host
  (common default: ~208KB) — worth raising via `sysctl` on a real deployment
  if you see high loss on bursty sends despite low idle RTT, since a small
  kernel buffer overflows (and silently drops) under a burst that a slower
  receiver hasn't drained yet.
- **External endpoint** / **Peer endpoint** — each side's public `ip:port`.
- **Hole punch success** / **Connection established** — the first packet
  from the peer arrived; direct UDP is working.
- **Direct P2P verified** — compares the address the first packet actually
  arrived from against what the coordinator reported. This is the answer to
  "are we sure this isn't secretly going through something else": the
  coordinator has no UDP-facing endpoint at all, so it structurally cannot
  relay this traffic regardless of what this check says; a `False` here
  just means the NAT remapped the port between registering and the first
  packet arriving (common, harmless, the benchmark still runs over the
  newly-observed direct address).

### Latency and loss

- **Idle RTT** — round-trip probes (200 by default, 50ms apart) sent before
  any bulk traffic. Reports average/P95/P99 RTT, jitter (mean absolute RTT
  delta between consecutive probes), and loss %. This is your baseline.
- **Loaded RTT** (in any streaming test) — the same kind of probe, but sent
  every 100ms *while* a stream is saturating the link. Compare directly
  against Idle RTT: a big gap means the link (or one side's CPU) is
  queueing/contending under load, which matters for latency-sensitive
  token-by-token traffic sharing the link with bulk transfers.

### Throughput and transfers

- **Transfer benchmark** / **Payload benchmark** / **Payload sweep** — a
  fixed amount of data (1/10/100MB, or 256KB-4MB "hidden-state-sized"
  payloads, or the full 256KB-32MB sweep) sent as `CHUNK_PAYLOAD`-sized (1200
  byte) UDP packets. For each: **throughput** (goodput in Mbps, computed from
  bytes the *receiver* actually confirmed getting, over the *full* elapsed
  time including that confirmation round trip — not just how fast the
  sender could push bytes into a socket), **loss %**, **completion
  latency**, packet count, **dup/ooo** (duplicate / out-of-order packets —
  should be ~0 for anything not genuinely reordered in transit), and
  **reassembly time** (span between the first and last chunk arriving at
  the receiver — a proxy for how "bursty vs. smooth" the delivery was, i.e.
  fragmentation/reassembly efficiency for that payload size).
- Both peers run the **same** transfer benchmark sequence on a shared
  `barrier()`-synchronized schedule, so — by design — both directions'
  traffic overlaps for the `default`/`payload-sweep` experiments. That's
  deliberate: it's what happens in real bidirectional inference traffic. If
  you want a *clean, uncontended* one-way number instead, use
  `--experiment long-streaming`, which has the two peers take turns (only
  one direction sends at a time).
- **Bandwidth-delay product** — computed from the best observed throughput
  and the idle RTT: `bandwidth × RTT` gives the bytes that should be
  in-flight to keep the link fully utilized, converted into a recommended
  packet window. Useful if you're designing an application-level flow
  control / windowing scheme on top of this.

### Streaming tests (`long-streaming`, `long-duplex`, `loaded-latency`, `multi-stream`, `fast-path-compare`, and the `default` experiment's streaming piece)

- **send**: avg/peak/min Mbps this side *attempted* to send, from local
  instrumentation — always exact, no round trip needed. `(peer confirmed
  X MB)` is the receiver's own tally, fetched via one final report request
  — the true goodput for what this side sent.
- **recv**: avg/peak/min Mbps *actually received* here, sampled every
  ~1s (first interval discarded — it partly measures barrier/scheduling
  skew before data starts flowing, which would otherwise show up as a
  spuriously low "min"). Always a direct, local, exact measurement.
- **loss / dup / ooo / late** on the receive side come from `StreamReceiver`,
  which tracks every sequence number seen: **loss** = gaps in the sequence
  space; **duplicates** = the exact same sequence number seen twice;
  **out-of-order** = a packet whose sequence number is *not* a new
  high-water mark (something higher already arrived first) but is still
  within a recent window; **late** = the same thing but so far behind the
  high-water mark it's outside that window — a very stale, likely
  meaningless-to-reorder-against arrival.
- **Multi-stream scaling** — aggregate throughput at 1/2/4/8/16 concurrent
  streams over the *same* socket, plus a `NxN=...x` scaling factor relative
  to 1 stream. If throughput does **not** scale with stream count, that's
  evidence the ceiling is per-packet Python/asyncio processing overhead on
  a single event loop thread, not available network bandwidth (more
  concurrent streams doesn't create more CPU).
- **Python overhead check** (`fast-path-compare`) — runs the identical
  duplex stream twice: once through the normal path (`pack_data()`
  allocates a new bytes object per packet, sent via the asyncio transport),
  once through a fast path (one preallocated buffer, only the sequence
  number patched in place via `struct.pack_into`, sent directly on the raw
  socket bypassing the transport wrapper). If fast-path throughput is
  meaningfully higher (the tool flags >1.3x as a rule of thumb) with lower
  CPU, **Python's own per-packet overhead is a real constraint** worth
  addressing (batching, a compiled extension, multiple processes). If it's
  about the same, the ceiling is elsewhere — the network path, the
  receiver's own processing, or CPU contention with whatever else is
  running on either host.

### CPU and timing

- **CPU utilization** — sampled every 0.5s via `getrusage()` during a
  phase: user%, system%, overall%, and peak (the single busiest 0.5s
  window). High overall with throughput far below line rate points at CPU
  as the bottleneck (Python-level or otherwise); low CPU with the same
  throughput points at the network or the receiver.
- **Time breakdown** (transfer/payload phase) — cumulative time this
  process spent inside `sendto()` calls, inside the `datagram_received()`
  callback, and building outgoing packets (`serializing`), versus
  `idle/asyncio` (everything else: `await`ing pacing sleeps, waiting on the
  event loop). For small one-shot transfers this is normally dominated by
  `idle` (the intentional pacing gaps that prevent overflowing the
  receiver's socket buffer) — that's expected, not a problem.

### NAT behavior

- **NAT keepalive test** (`default` experiment only) — idles for 10s, 30s,
  60s, 120s (`--quick`: 2/3/5/8s) with *no* traffic in between, then sends a
  probe (retried up to 3 times before declaring failure, since one dropped
  probe isn't proof the mapping expired) to see if the NAT mapping is still
  open. This is the number that tells you how often you need background
  keepalive traffic in a real deployment.
- **NAT rebinding detected** — two independent mechanisms, both always
  active: (1) *passive* — if a packet ever arrives from an address other
  than the one currently on file for the peer, that's logged immediately as
  `(peer side)` and the new address is adopted so the benchmark keeps
  working through a mid-session rebind; (2) *active* — every 20s, each peer
  asks the other "what address did you last see me from?" (piggybacked on
  the same UDP socket, no extra STUN calls) and compares the answer to what
  it expects, logging `(own side)` on a change. The final report's count is
  the total across both mechanisms.

Both peers print `Waiting for peer to finish its own benchmarks...` before
their summary. Whichever side finishes first waits (up to 2 minutes,
announcing itself every second over the same direct channel) rather than
closing its socket and exiting — otherwise the slower peer's last probe
would fail simply because the other process already quit, which would look
exactly like (but is not) a real NAT mapping timeout.

## Is this measuring the network or Python?

Three pieces of evidence, all in the final report, answer this directly for
any given run:

1. **CPU utilization** during a streaming/duplex phase. If it's pegged near
   100% while throughput is well below the link's real capacity, Python is
   in the way.
2. **`fast-path-compare`**. If the raw-socket/preallocated-buffer path
   isn't meaningfully faster than the normal path, per-packet Python
   overhead isn't the ceiling — something else is (receiver processing,
   actual link bandwidth, or CPU shared with other processes on the host).
3. **Multi-stream scaling**. Throughput that doesn't grow with concurrent
   stream count on the same socket/event-loop/CPU core is further evidence
   of a single-threaded processing ceiling rather than a bandwidth one.

On a shared, CPU-constrained box (e.g. testing both peers as two processes
on the *same* small VM, which is not how it's meant to be deployed but is
how this tool was validated during development — see below), expect CPU
contention to dominate: two processes fighting over 1-2 cores while both
try to saturate a loopback link will show real loss and CPU near 50-100%
that has nothing to do with actual network capacity. Run peer A and peer B
on the two real separate servers to get numbers that reflect the actual
network path between them.

## How this was validated

Both `peer.py` and `signaling_server.py` were run end-to-end, locally
(loopback, two peer processes against a local signaling server), through
**every** `--experiment` mode before being handed over. That process caught
and fixed three real bugs, in case any of these numbers looked odd in a
previous run:

- `StreamReceiver` used to get its "next expected sequence number" pointer
  permanently stuck the first time *any* packet was lost, after which every
  subsequent (perfectly in-order) packet was miscounted as out-of-order —
  producing nonsense numbers like tens of thousands of "reordered" packets
  on a loopback link. Reordering is now judged purely by whether a packet's
  sequence number is a new high-water mark, independent of any earlier gap.
- The receiving side of a streaming test used to delete its tracking state
  immediately after its own local barrier, which could race ahead of the
  sender's final "what did you actually get" request and return a phantom
  empty report. Fixed by not deleting it.
- `--fast-path` initially tried to get the raw socket via
  `transport.get_extra_info("socket")`, which returns asyncio's
  intentionally-restricted `TransportSocket` wrapper (no `sendto()`) —
  crashed immediately. Fixed by keeping a direct reference to the real
  socket object from `main()`.

If hole punching fails outright (`Hole punch FAILED` printed, no benchmark
runs), one or both NATs are likely symmetric (destination-dependent
mapping) rather than cone NAT — see the earlier NAT-type investigation in
this environment. That's the case where a TURN/relay server would actually
be required; this tool intentionally has no relay fallback, so a failure
here is itself the useful signal.
