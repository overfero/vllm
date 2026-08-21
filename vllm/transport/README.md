# Transport abstraction — scope, design, and status

This phase proves one thing: vLLM's worker-to-worker communication can be
expressed behind a swappable interface, and an existing NAT-hole-punched
UDP transport can sit behind that interface as a real alternative to plain
TCP. It deliberately does **not** touch model execution, CUDA, NCCL, or
tensor/pipeline-parallel scheduling — see "Constraints honored" below.

## 1. Modified files

### New files (the transport abstraction itself)

| File | Why it exists | Responsibility |
|---|---|---|
| `vllm/transport/base.py` | Defines the contract everything else implements | `Transport` ABC (`connect`/`send`/`recv`/`close`) and `TransportConfig`, the one config shape both backends read from |
| `vllm/transport/tcp_transport.py` | A baseline backend to prove the abstraction is genuinely swappable | `TCPTransport`: length-prefixed framing over a plain blocking socket, one listener/one connector |
| `vllm/transport/udp_transport.py` | Adapts the existing, already-proven UDP hole-punch transport | `UDPTransport`: imports `peer.py` unmodified (see §"Reuse, not duplication"), runs it on a background asyncio loop, adds message framing + a reliability layer on top |
| `vllm/transport/factory.py` | The one place backend selection happens | `get_transport(backend=None)`, resolving explicit arg → `VLLM_TRANSPORT` env var → `"tcp"` default. Nothing else should ever branch on backend name. |
| `vllm/transport/__init__.py` | Public surface | Re-exports `Transport`, `TransportConfig`, `get_transport` |
| `vllm/transport/README.md` | This file | Scope, design rationale, and the deliverables below |
| `tests/transport/_common.py` | Shared test infrastructure | Spins up a fresh signaling-server instance per UDP test run, builds `TransportConfig` pairs, spawns worker processes |
| `tests/transport/test1_hello_world.py` … `test5_concurrent.py` | The five required communication tests | Each runs unmodified against both `--transport tcp` and `--transport udp` |

### Existing vLLM files touched (minimal, additive)

| File | What changed | Why |
|---|---|---|
| `vllm/envs.py` | Added `VLLM_TRANSPORT: Literal["tcp","udp","quic"] = "tcp"` (type stub + `env_with_choices` runtime entry, next to `VLLM_WORKER_MULTIPROC_METHOD`) | Config-driven selection, per the requirement — follows the file's own existing pattern exactly, 4 lines total |
| `vllm/config/parallel.py` | Added `TransportBackend = Literal["tcp","udp","quic"]` alias and `transport_backend: TransportBackend = "tcp"` field (+docstring) on `ParallelConfig`, next to `master_addr`/`master_port` | Makes the backend a first-class, documented config field alongside the other distributed-runtime settings it will eventually sit next to |
| `vllm/engine/arg_utils.py` | Mirrored the field on `EngineArgs`, added `TransportBackend` to the import block, added one `parallel_group.add_argument("--transport", ...)` line, threaded `transport_backend=self.transport_backend` into the `ParallelConfig(...)` construction in `create_engine_config` | Gives `--transport {tcp,udp}` as a real CLI flag, following the exact `--master-addr`/`--master-port` precedent (same file, same pattern, same `get_kwargs()` auto-generation) |

**Nothing else in vLLM was touched.** No changes to `vllm/distributed/`,
`vllm/worker/`, `vllm/v1/worker/`, `vllm/executor/`, or any model/attention/
scheduler code.

### Verification status of the config/CLI edits

The `vllm/config/parallel.py` and `vllm/engine/arg_utils.py` edits are
syntax-verified (`py_compile`) and pattern-matched against the existing
`master_addr`/`master_port` precedent, but **not runtime-exercised** in
this environment: importing `vllm.config` transitively imports
`vllm.distributed.parallel_state`, which fails at import time on this
torch build (`torch==2.6.0+cu124`) with

```
ValueError: infer_schema(func): Parameter output_shape has unsupported
type list[int] ...
```

inside `direct_register_custom_op` — a pre-existing torch/vLLM version
incompatibility on `main`, unrelated to anything in this change (reproduces
on a clean checkout before any of these edits). `vllm.envs` and the new
`vllm.transport` package both import and run cleanly on their own (verified
directly, see the test runs below) since neither depends on that chain.

## 2. Dependency graph — before

```
Worker (vllm/v1/worker/gpu_worker.py)
    │
    ├─ init_device()                       # fuses device + distributed init, one method
    │     └─ init_worker_distributed_environment()
    │           ├─ init_distributed_environment()
    │           │     └─ torch.distributed.init_process_group(
    │           │            init_method="tcp://{master_addr}:{master_port}")
    │           │                 └─ TCPStore  (rendezvous / control plane)
    │           └─ ensure_model_parallel_initialized(tp_size, pp_size, ...)
    │                 └─ GroupCoordinator (per parallel group: TP, PP, DP, world)
    │                       └─ device_communicator  (picked by platform, not swappable)
    │                             ├─ CudaCommunicator → pynccl (NCCL) ──▶ Socket
    │                             │                   → torch.distributed.send/recv (gloo) ──▶ Socket
    │                             └─ CpuCommunicator  → torch.distributed.send/recv (gloo) ──▶ Socket
    │
    └─ model forward pass → get_pp_group().send_tensor_dict() / irecv_tensor_dict()
                                  (data-plane P2P for pipeline-parallel intermediate tensors)
```

Two things worth being precise about, since they change what "swappable"
can honestly mean here:

- **Rendezvous vs. data plane are different layers.** Only the rendezvous
  URI (`tcp://host:port`, literally hardcoded in
  `vllm/utils/network_utils.py:get_tcp_uri`) is a plain, replaceable
  string. The actual tensor `send`/`recv` traffic goes through NCCL
  (GPU-GPU) or gloo, both of which manage their own internal sockets
  opaquely to vLLM.
- **"Transport" isn't an axis the current code models at all.**
  `GroupCoordinator` picks its communicator by *device platform*
  (Cuda/Cpu/Xpu), not by a transport choice — TCP vs UDP doesn't exist as
  a concept anywhere in `vllm/distributed/` today.

## 3. Dependency graph — after

```
Worker (unchanged)
    │
    ├─ init_device()                                    (unchanged - still fused, untouched)
    │     └─ ... exactly as above ...
    │
    └─ [NEW, parallel/independent] vllm.transport
          │
          get_transport()  ◀── VLLM_TRANSPORT env var / --transport CLI flag
          │        (the one place selection happens - vllm/transport/factory.py)
          │
          ├─ TCPTransport ──────────────▶ plain socket, length-prefixed framing
          │
          └─ UDPTransport ──▶ peer.py (existing, UNMODIFIED)
                                  ├─ STUN, NAT hole punch, signaling client
                                  ├─ PeerProtocol (subclassed, not modified)
                                  └─ reliability/reassembly (new, in the adapter)
```

The new tree is **not yet wired into** the "before" tree — `transport_backend`
is a real, selectable config value that reaches `ParallelConfig`, but nothing
in `vllm/distributed/` reads it yet. That's the honest state: the
abstraction exists, is tested, and is swappable *on its own*; connecting it
to the real data plane is explicitly deferred (§5).

### Reuse, not duplication

`udp_transport.py` never reimplements hole punching, STUN, signaling, or
the wire protocol. It:
1. Locates the existing transport's directory (`VLLM_UDP_TRANSPORT_DIR`,
   defaulting to the `udp_holepunch/` sibling directory at this repo's
   root) and imports `peer.py` as a library.
2. Subclasses `peer.PeerProtocol` (`_AdapterProtocol`), adding handling for
   three new wire tags (`M` message chunk, `Q`/`A` status query/answer)
   that the original class doesn't recognize and therefore silently
   ignores — strictly additive, the original dispatch is called first via
   `super()._handle_datagram()` and always runs unchanged.
3. Reuses `peer.stun_get_mapped_address`, `peer.register`, `peer.wait_for_peer`,
   `peer.barrier`, `peer.punch_loop`, `peer.CHUNK_PAYLOAD`,
   `peer.SOCKET_BUFFER_REQUEST`, and `peer.PACING_GAP_SECONDS` directly.

## 4. What still depends directly on NCCL / CUDA / the distributed runtime

Everything in the actual execution path, unconditionally:

- **`vllm/distributed/device_communicators/cuda_communicator.py`** —
  `CudaCommunicator.send`/`recv` call `self.pynccl_comm.send/recv` (NCCL)
  directly, falling back to `torch.distributed.send/recv` only if pynccl
  is disabled. No transport indirection.
- **`vllm/distributed/parallel_state.py`** — `GroupCoordinator` is
  constructed against `torch.distributed` process groups
  (`self.cpu_group`, `self.device_group`) at `init_distributed_environment`
  time; `send_tensor_dict`/`recv_tensor_dict`/`broadcast_tensor_dict` all
  assume a live process group.
- **`vllm/v1/worker/gpu_worker.py` / `gpu_model_runner.py`** — the actual
  pipeline-parallel call sites (`get_pp_group().isend_tensor_dict(...)`,
  `irecv_tensor_dict(...)`, `send_tensor_dict(...)`,
  `broadcast_tensor_dict(...)`) are unconditional; there is no branch point
  today where a non-NCCL transport could intercept them.
- **`init_worker_distributed_environment` → `init_device()`** — CUDA
  device selection and NCCL/gloo process-group construction happen in the
  same method, with no separable "bring up communication" step independent
  of "bring up the GPU".

Independent of that: **`StatelessProcessGroup`**
(`vllm/distributed/utils.py:199`) is a CPU-only, NCCL-free metadata channel
already built on a raw `torch._C._distributed_c10d.Store`
(`send_obj`/`recv_obj`/`barrier` via `store.set`/`store.get`). It's the one
piece of the current distributed stack that's already decoupled from
NCCL/CUDA — and the most realistic place a `vllm.transport` backend could
plug in without touching the data plane at all.

## 5. Next components to modify for pipeline-parallel to use UDP transport

**Not implemented in this phase, listed only.** In rough dependency order:

1. **`vllm/distributed/utils.py` — `StatelessProcessGroup`**: teach it (or
   an alternative next to it) to route its `send_obj`/`recv_obj`/`barrier`
   calls through `vllm.transport.Transport` instead of directly through a
   `Store`. This is the lowest-risk integration point: control-plane only,
   already CPU/NCCL-free, and it's what a first real "UDP for metadata"
   milestone would target.
2. **`vllm/distributed/device_communicators/base_device_communicator.py`**:
   introduce a transport-backed communicator (e.g. `TransportCommunicator`)
   implementing the same `send`/`recv` interface as `CudaCommunicator`/
   `CpuCommunicator`, so `GroupCoordinator` has a non-NCCL, non-gloo option
   for point-to-point traffic. Needs a tensor-serialization story (today's
   `send_tensor_dict` metadata format assumes a torch process group is
   available for the accompanying collective ops).
3. **`vllm/distributed/parallel_state.py` — `GroupCoordinator.__init__` /
   `init_distributed_environment`**: today the communicator class is
   selected purely by device platform. Adding transport as a second,
   orthogonal selection axis means threading `transport_backend` through
   group construction, not just device type.
4. **`vllm/v1/worker/gpu_worker.py` — `init_worker_distributed_environment` /
   `init_device()`**: separate "bring up the communication layer" from
   "bring up the CUDA device", so a transport can be established before
   (or independent of) device/model init — currently these are fused in
   one method with no seam.
5. **Tensor payload framing**: `send_tensor_dict`/`recv_tensor_dict` today
   move raw GPU tensors via NCCL/gloo primitives that understand device
   memory directly. A transport-backed path would need explicit
   host-memory staging (D2H/H2D copies) since `vllm.transport.Transport`
   only ever moves `bytes` — a real performance-relevant design decision,
   not a mechanical one, and explicitly out of scope for "communication
   only, no performance optimization" in this phase.

None of the above were touched. This phase stops at: the abstraction
exists, is config-selectable, and is proven correct end-to-end
(hello/world, byte-perfect payloads up to 16MB, 1000-ping RTT/jitter/loss,
100MB streaming, 4-way concurrent — all passing for both backends).

## Constraints honored

No HuggingFace model load, no `torch.cuda` init, no NCCL init, no
attention/KV-cache/scheduler/TP/PP execution anywhere in this phase's code
or tests. `tests/transport/*.py` import only `vllm.transport` (which
imports `vllm.envs`, confirmed CUDA/model-free) plus stdlib — never
`vllm.config`, `vllm.engine`, or `vllm.distributed`.

## Test results (this environment, loopback, both peers on one machine)

All 5 tests × 2 backends = 10 runs, all passing:

| Test | TCP | UDP |
|---|---|---|
| 1: hello/world | PASS | PASS |
| 2: payload sizes (1KB–16MB, byte-perfect) | PASS (16MB @ 1490 Mbps) | PASS (16MB @ 247 Mbps, 536ms) |
| 3: 1000 ping-pong (RTT/jitter/loss) | PASS (avg 0.099ms, 0% loss) | PASS (avg 0.521ms, 0% loss) |
| 4: 100MB streaming | PASS (24.3 Gbps, 0 retransmits) | PASS (280 Mbps, 2.99s) |
| 5: 4-worker concurrent, 2 pairs | PASS (no deadlock/cross-talk) | PASS (no deadlock/cross-talk) |

UDP numbers reflect this single-sandbox loopback environment (both
"workers" sharing CPU), not a real two-machine network path — see the
existing transport's own README for that caveat in detail, and rerun these
same scripts across two real hosts (`VLLM_TRANSPORT=udp`, real
`--signaling-url`) for numbers that mean something about actual network
capacity.

## Bugs found and fixed while validating this integration

Testing against real payload sizes (not just tiny hello/world) surfaced
two real bugs in the new adapter code (the imported `peer.py` was
untouched and not implicated):

1. **No reliability layer initially** — a naive one-shot "send all chunks,
   done" implementation meant a single lost UDP datagram out of thousands
   left the receiver's `recv()` blocked forever. Fixed with a batched
   (~600KB/batch) send-then-confirm-then-resend-if-short cycle, so backlog
   size (and therefore confirmation latency) stays bounded regardless of
   total message size.
2. **Timeout miscalibration** — the first version of that fix used a flat,
   multi-second confirmation timeout sized for large-message backlog
   processing, which made every small message (e.g. a 4-byte ping) pay
   worst-case latency. Fixed with an RTT-scaled timeout (200ms first
   attempt, exponential backoff on retry) so the common case is fast and
   only genuine loss/backlog pays the longer wait.

---

# Phase 2: real tensors over the transport abstraction

Builds directly on phase 1 above. Same constraints (no HF model, no CUDA
init, no NCCL, no torch.distributed, no Ray, no attention/KV-cache/
scheduler init, CPU only) — this phase only adds `torch.Tensor` on top of
the already-proven `bytes` transport. `TCPTransport`/`UDPTransport`
themselves are **still untouched**.

## 1. Modified files

| File | Why it exists | Responsibility |
|---|---|---|
| `vllm/transport/tensor.py` | The only new production file this phase needed | `serialize_tensor`/`deserialize_tensor` (dtype+shape metadata → raw bytes, and back) and `TransportProcessGroup` (`send_tensor`/`recv_tensor`, built on `Transport.send`/`recv`) |
| `tests/transport/test6_tensor_small.py` … `test10_tensor_concurrent.py` | The five required tensor tests | Same "identical worker code across `--transport tcp/udp`" pattern as phase 1's tests |

Nothing else changed. `base.py`, `tcp_transport.py`, `udp_transport.py`,
`factory.py` are byte-for-byte what phase 1 left them — this phase only
adds a layer *above* `Transport`, never inside it.

`vllm/transport/__init__.py` was deliberately **not** changed to re-export
`tensor.py`'s symbols: the base package (`base`/`tcp_transport`/
`udp_transport`/`factory`) imports zero third-party dependencies, and
`tensor.py` is the first file in this package that needs `torch`. Keeping
`import vllm.transport` torch-free and making tensor support an explicit
`from vllm.transport.tensor import ...` opt-in preserves that property for
any caller that only needs raw byte transport.

### Integration point chosen: new `TransportProcessGroup`, not `StatelessProcessGroup`

The task's preferred order was "use `StatelessProcessGroup` if possible,
otherwise a new lightweight `TransportProcessGroup`." `StatelessProcessGroup`
was evaluated and rejected for this role: its `send`/`recv`/`send_obj`/
`recv_obj` (`vllm/distributed/utils.py:227-314`) all go through
`self.store.set`/`self.store.get` on a `torch._C._distributed_c10d.Store`
(concretely a `TCPStore`, see `create_tcp_store` in the same file) — i.e.
it already *has* its own data plane, a raw TCP-based KV store, completely
independent of `vllm.transport`. Routing tensors through it would move them
over that store's socket, not over `TCPTransport`/`UDPTransport` — which
would fail the actual goal ("prove tensors can travel through the Transport
abstraction"). `StatelessProcessGroup` remains the right *future*
integration point for control-plane traffic (§5 below still lists it
first), but not for this phase's tensor data plane. `TransportProcessGroup`
is therefore new, minimal (46 lines), and lives in `vllm/transport/` rather
than `vllm/distributed/` — it doesn't touch `GroupCoordinator` at all.

## 2. Current architecture (this phase)

```
Worker
    │
    ├─ [unchanged] init_device() → NCCL/gloo process groups → GroupCoordinator
    │              (real inference path - completely untouched, see phase 1 §4)
    │
    └─ [NEW] TransportProcessGroup(transport)
              │
              ├─ send_tensor(tensor)
              │     tensor ──serialize_tensor()──▶ metadata (dtype,shape) + raw bytes
              │                                          │
              │                                   transport.send(bytes)
              │                                          │
              │                              TCPTransport  or  UDPTransport
              │
              └─ recv_tensor()
                    transport.recv() ──▶ bytes ──deserialize_tensor()──▶ tensor
                                                  (dtype/shape restored, torch.equal-exact)
```

`serialize_tensor`/`deserialize_tensor` (`vllm/transport/tensor.py`):
dtype-agnostic — the payload is `tensor.contiguous().view(torch.uint8)`
reinterpreted as raw bytes, not a per-dtype numpy conversion table. This is
what makes `bfloat16` (which numpy cannot represent) round-trip correctly
alongside `float32`/`float16`/`int32`/`int64`/`bool` with the same code
path. Metadata (`dtype` name + `shape`) is a small length-prefixed JSON
header in front of the raw bytes, so one `transport.send()` call carries
one complete tensor.

## 3. NCCL / CUDA / distributed-runtime dependencies

Unchanged from phase 1 §4 — nothing in this phase touches
`cuda_communicator.py`, `GroupCoordinator`, or `gpu_worker.py`. Restated
briefly: `GroupCoordinator.send`/`recv` (`vllm/distributed/parallel_state.py:1209-1223`)
unconditionally delegate to `self.device_communicator.send/recv`
(`CudaCommunicator` → pynccl/NCCL, or `CpuCommunicator` → gloo), and
`self.device_communicator` only exists because `GroupCoordinator.__init__`
already required a live `torch.distributed` process group
(`self.device_group`/`self.cpu_group`, built via `torch.distributed.split_group`/
`new_group`) to construct it. `TransportProcessGroup` was added *next to*
this, not inside it — see §4.

## 4. What exactly prevents replacing `GroupCoordinator` with the transport interface

Three separate couplings, not one:

1. **Construction-time coupling.** `GroupCoordinator.__init__`
   (`vllm/distributed/parallel_state.py:380` onward) takes a list of global
   ranks and immediately builds `torch.distributed` process groups from
   them (`self_device_group`/`self_cpu_group` via `torch.distributed.split_group`).
   There is no code path to construct a `GroupCoordinator` without a
   process group already existing — `transport.connect()`'s peer-to-peer
   model (one `self_id`, one `peer_id`) has no equivalent of "rank" or
   "world_size" to build one from.
2. **Selection-axis coupling.** The communicator inside a `GroupCoordinator`
   is chosen by *device platform* (`CudaCommunicator` vs `CpuCommunicator`
   vs Xpu/etc., `vllm/distributed/parallel_state.py:502-520`), not by a
   transport choice. "Which transport" isn't a variable `GroupCoordinator`
   knows how to branch on anywhere — adding one means a second, orthogonal
   selection axis through the same construction path (this is exactly item
   3 in §5 below).
3. **Data-shape coupling.** `send`/`recv` move raw device-memory tensors
   through NCCL/gloo primitives that understand GPU memory pointers
   directly. `Transport.send`/`recv` only ever move `bytes` on the host.
   Even with a transport-backed communicator, every call would need
   explicit D2H/H2D staging that NCCL/gloo don't need — a real design
   decision (buffer reuse, pinned memory, overlap with compute), not a
   drop-in swap. This phase's `serialize_tensor`/`deserialize_tensor` do the
   CPU-only version of this (`tensor.contiguous().view(torch.uint8).numpy()`)
   with zero attempt at optimizing it, per the "correctness first" constraint.

In short: `GroupCoordinator` isn't a `send`/`recv` function with a
transport plugged in underneath that could be swapped — the transport
choice would need to reach all the way back to *how the group itself gets
constructed*, before any tensor ever moves.

## 5. Smallest change to make one pipeline-parallel activation tensor use the new transport

**Not implemented — identified only,** per the task. Smallest viable slice,
in order:

1. Pick one specific call site: `vllm/v1/worker/gpu_worker.py`'s use of
   `get_pp_group().send_tensor_dict(...)` / `recv_tensor_dict(...)` for the
   single-tensor case (skip the multi-tensor/dict metadata path entirely
   for a first slice — one `torch.Tensor` in, one out).
2. In `GroupCoordinator` (`vllm/distributed/parallel_state.py`), add a
   `transport: Transport | None = None` constructor field, populated only
   when `self.rank_in_group` has exactly 2 members and
   `parallel_config.transport_backend != None` — i.e. additive, opt-in,
   never touching the existing `device_communicator` path for any group
   that doesn't request it.
3. Add one conditional branch in `GroupCoordinator.send`/`.recv`
   (`parallel_state.py:1209-1223`): if `self.transport is not None`, call
   `TransportProcessGroup(self.transport).send_tensor(tensor)` /
   `.recv_tensor()` (staging D2H/H2D around it if `tensor.is_cuda`) instead
   of `self.device_communicator.send/recv`. This is the *only* branch point
   in the entire call chain — everywhere else (`gpu_worker.py`,
   `gpu_model_runner.py`) already calls through `GroupCoordinator`, so nothing
   upstream needs to change.
4. `transport.connect()` still needs a `self_id`/`peer_id`/signaling
   endpoint per PP link, which today's rank-based `GroupCoordinator`
   construction doesn't produce — bridging "PP rank N and N+1" to
   "transport peer IDs" is the one genuinely new piece of plumbing, not a
   mechanical change.

Deliberately excluded from this slice (real follow-on work): multi-tensor
`send_tensor_dict`, tensor-parallel (>2 ranks — `TransportProcessGroup` is
point-to-point only), and any performance work (pinned-memory staging,
overlapping D2H copy with compute, batching).

## 6. Test results — tensor phase (loopback, both peers on one machine)

All 5 tensor tests × 2 backends = 10 runs, all passing, no code changes
needed to phase 1's transport/reliability layer:

| Test | TCP | UDP |
|---|---|---|
| 6: small tensors (16–1024 elem × 6 dtypes incl. bfloat16/bool) | PASS (24/24 exact) | PASS (24/24 exact) |
| 7: large tensors (1/4/16/64 MB, timing breakdown) | PASS (64MB: 411ms total, 1.3 Gbps) | PASS (64MB: 2.24s total, 240 Mbps) |
| 8: streaming 1000 tensors (~16.4MB total) | PASS (0.14s, 910 Mbps, 0 lost/corrupted) | PASS (0.81s, 162 Mbps, 0 lost/corrupted) |
| 9: bidirectional simultaneous (1MB each way, concurrent send+recv threads) | PASS (9ms) | PASS (~40ms) |
| 10: 4-worker concurrent, 2 pairs | PASS (no deadlock/cross-talk) | PASS (no deadlock/cross-talk, connect ~5.1s/worker) |

Test 7's per-size breakdown (serialize / transport / deserialize /
bandwidth / CPU) is printed by the test itself — see its output for the
full table; both transports show serialization/deserialization cost
scaling roughly linearly with tensor size and dominated by `transport_s`
for UDP (chunking + confirm/resend overhead) vs. dominated by kernel-buffer
copy for TCP.

## Bugs found this phase

None. Reusing `Transport.send`/`recv` as an opaque `bytes` channel meant
phase 1's reliability layer (batched confirm/resend for UDP, length-prefixed
framing for TCP) was already sufficient for arbitrarily large serialized
tensors (tested up to 64MB) — the only new code this phase added
(`tensor.py`) is pure serialization logic with no networking of its own, so
there was no new class of bug to hit.

---

# Phase 3: a real vLLM pipeline-parallel send/recv path, intercepted

Builds on phases 1-2. Same hard constraints (no HF model, no CUDA init, no
NCCL init, no torch.distributed init, no Ray, no attention/KV-cache/
scheduler init) — this phase modifies `vllm/distributed/parallel_state.py`
itself for the first time, but only ever *constructs* it in a way that
never calls `torch.distributed.init_process_group`, never touches CUDA,
and never creates an NCCL communicator. See §"How the real class is
constructed here" below for exactly how that's possible.

## 1. Modified files

| File | What changed | Why |
|---|---|---|
| `vllm/distributed/parallel_state.py` | Added a `transport: Transport \| None = None` class-level default + `__init__` parameter on `GroupCoordinator`, and one `if self.transport is not None: ...` branch each at the top of `.send()` and `.recv()` | The actual interception point — see §2. This is the only production file this phase touched. |
| `tests/transport/_pipeline_shim.py` | New (test-only) | Makes the real `GroupCoordinator` importable/constructable in this sandbox without `torch.distributed`/CUDA/NCCL — see §"How the real class is constructed here" |
| `tests/transport/test11_pipeline_single.py` … `test15_pipeline_three_stage.py` | New | The five required pipeline tests |

`vllm/transport/tensor.py`, `base.py`, `tcp_transport.py`, `udp_transport.py`,
`factory.py` are untouched — this phase's only new *production* code is the
6-line branch inside `GroupCoordinator`; everything else it needed
(`TransportProcessGroup`) already existed from phase 2.

### The diff, in full

```python
# class-level default (so subclasses that build instances without calling
# this __init__, e.g. StatelessGroupCoordinator, still safely default to
# the existing NCCL/gloo path)
transport: "Transport | None" = None

def __init__(self, ..., transport: "Transport | None" = None):
    ...
    self.transport = transport
    ...

def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
    if self.transport is not None:
        from vllm.transport.tensor import TransportProcessGroup
        TransportProcessGroup(self.transport).send_tensor(tensor)
        return
    if self.device_communicator is None:
        raise ValueError("No device communicator found")
    self.device_communicator.send(tensor, dst)

def recv(self, size, dtype, src=None) -> torch.Tensor:
    if self.transport is not None:
        from vllm.transport.tensor import TransportProcessGroup
        tensor, _stats = TransportProcessGroup(self.transport).recv_tensor()
        return tensor
    if self.device_communicator is None:
        raise ValueError("No device communicator found")
    return self.device_communicator.recv(size, dtype, src)
```

`TransportProcessGroup` is imported lazily, inside the branch, so
`import vllm.distributed.parallel_state` never pulls in `torch` twice or
creates a hard dependency on `vllm.transport` for any group that doesn't
opt in. Every existing call site, every existing group, keeps behaving
exactly as before: `transport` defaults to `None` everywhere it wasn't
explicitly set, so the new branch is dead code unless something
deliberately passes a `Transport`.

## 2. Exactly where the existing pipeline path was intercepted

`GroupCoordinator.send()` / `.recv()` (`vllm/distributed/parallel_state.py:1219-1243`
after this change). This is not a hypothetical or convenience method — it's
the same single-tensor point-to-point API that:

- **vLLM's own test suite exercises directly** for pipeline parallelism:
  `tests/distributed/test_comm_ops.py::send_recv_test_worker` calls
  `get_pp_group().send(test_tensor)` / `get_pp_group().recv(size, dtype=...)`
  under real multi-GPU + Ray + torch.distributed.
- **A real executor backend calls in production**:
  `vllm/distributed/device_communicators/ray_communicator.py`'s
  `RayPPCommunicator` (used for pipeline parallelism under Ray Compiled
  Graph) wraps `get_pp_group().device_communicator` and calls
  `.send(buf, peer_rank)` / `.recv(size, dtype, src=peer_rank)` on it for
  every activation hop — one layer below where this phase's branch sits
  (see §3 for why the branch is in `GroupCoordinator`, not
  `CudaCommunicator`, and what that means for the Ray path specifically).

The much more heavily-used **bulk** activation-passing API,
`send_tensor_dict`/`recv_tensor_dict` (what `gpu_model_runner.py` actually
calls today for multi-tensor pipeline-parallel activations), was
deliberately **not** touched — it calls `torch.distributed.isend`/`irecv`
directly or `self.device_communicator.send_tensor_dict`, never routing
through `.send()`/`.recv()`. Redirecting it was out of scope ("smallest
possible change", "one activation path"); see phase 2 §5 for what that
would additionally require (multi-tensor framing, dict metadata).

## 3. Which NCCL calls were bypassed

None are *removed* — `CudaCommunicator.send`/`.recv()`
(`vllm/distributed/device_communicators/cuda_communicator.py:532-558`,
which call `pynccl_comm.send`/`.recv()`) are completely unmodified. What's
bypassed is *reaching* them: when `GroupCoordinator.transport` is set, the
new branch in `GroupCoordinator.send`/`.recv()` returns before
`self.device_communicator.send/recv` is ever called, so
`pynccl_comm.send`/`.recv()` (and, in the CPU/gloo case,
`CpuCommunicator`'s `torch.distributed.send`/`recv`) never execute for that
call. This is a call-site bypass one layer up, not a change to NCCL's code
path itself — exactly the "smallest possible change" the task asked for.

## 4. Remaining NCCL entry points

Counting only `CudaCommunicator` (`cuda_communicator.py`), the class
`GroupCoordinator.device_communicator` actually is on a CUDA platform:

| Method | Bypassed by this phase? |
|---|---|
| `send` / `recv` | Only when `GroupCoordinator.transport` is set (this phase) |
| `all_reduce` | No |
| `all_gather` / `all_gatherv` | No |
| `reduce_scatter` / `reduce_scatterv` | No |
| `broadcast` | No |
| `batch_isend_irecv` | No |

**8 unconditional NCCL entry points remain** in `CudaCommunicator` alone,
untouched. On top of that, still fully NCCL/torch.distributed-dependent
and untouched by this phase:

- `send_tensor_dict` / `recv_tensor_dict` / `isend_tensor_dict` (§2 above) —
  the actual bulk PP activation path.
- `GroupCoordinator.all_reduce`/`all_gather`/`broadcast`/`gather`/
  `send_object`/`recv_object`/`barrier` — all still delegate to
  `device_communicator` or raw `torch.distributed.*` collectives, no
  `transport` branch added (out of scope: this phase is P2P send/recv
  only, not collectives).
- `GroupCoordinator.__init__` itself, for any group that doesn't pass
  `transport=` — still calls `torch.distributed.new_group`/`split_group`
  exactly as before.
- Every device-communicator constructor path (`CudaCommunicator.__init__`
  building a `PyNcclCommunicator`) — entirely unreached in this phase's
  tests, since no `GroupCoordinator` here was ever constructed with
  `use_device_communicator=True` (see §"How the real class is constructed
  here").

## 5. New architecture

```
Pipeline (Stage0 / Stage1 / Stage2 - dummy, no model)
    │
    ├─ stage.send(activation) / stage.recv(size, dtype)
    │        (GroupCoordinator.send/.recv - UNCHANGED SIGNATURE,
    │         the real method vLLM's PP code and test suite call)
    │
    └─ inside GroupCoordinator.send/.recv:
            │
            ├─ if self.transport is not None:        ◀── NEW, this phase
            │        TransportProcessGroup(self.transport)
            │              .send_tensor() / .recv_tensor()
            │              │
            │              ├─ TCPTransport
            │              └─ UDPTransport
            │
            └─ else (unchanged, existing path):
                     self.device_communicator.send/recv
                              │
                              ├─ CudaCommunicator → pynccl_comm ──▶ NCCL
                              └─ CpuCommunicator  → torch.distributed ──▶ gloo
```

## How the real class is constructed here

Two independent, pre-existing obstacles had to be worked around to
actually *run* this against the real `GroupCoordinator` class rather than a
reimplementation — both handled entirely inside
`tests/transport/_pipeline_shim.py`, zero additional vLLM source touched
beyond the §1 diff:

1. **Import-time failure, unrelated to this change.** `import
   vllm.distributed.parallel_state` fails on this sandbox's torch build
   (`2.6.0+cu124`) at a module-level `direct_register_custom_op(...)` call
   (line ~373) whose op signature isn't representable by this torch
   version's `infer_schema` — the surrounding code has its own `# TODO:
   Remove this once the pytorch fix ... gets released, in either 2.9.1 or
   2.10` comment acknowledging it. Same root cause phase 1 already hit and
   documented for `vllm.config`. The shim monkeypatches
   `vllm.utils.torch_utils.direct_register_custom_op` to a no-op *before*
   importing `parallel_state` (so the `from ... import
   direct_register_custom_op` inside it picks up the stub) - test-file-only,
   never touches vLLM source, and has no effect on `GroupCoordinator` itself
   since none of this phase's tests call the ops that registration would
   have provided.
2. **`GroupCoordinator.__init__` requires live `torch.distributed`.** It
   unconditionally calls `torch.distributed.get_rank()` and
   `torch.distributed.new_group(...)` — i.e. the normal constructor cannot
   run under this phase's "no torch.distributed init" constraint, by
   design (that's real, load-bearing vLLM behavior, not a bug). Since the
   new `.send()`/`.recv()` branch (§1) is the *first* thing those methods
   do and touches no other `self.*` attribute before returning,
   `object.__new__(GroupCoordinator)` (skipping `__init__` entirely) with
   only `.transport` set is sufficient to call the real, patched methods
   correctly. Verified directly: a bare instance's `.send()`/`.recv()`
   round-tripped a real tensor between two processes before any test file
   was written (see the smoke test in this phase's history).

Both workarounds are additive/isolated to the test harness; the production
diff in §1 works identically whether `GroupCoordinator` is constructed the
normal way (with `transport=` passed through the real constructor once a
caller wires it up — not done in this phase, see phase 2 §5 item 3) or via
this bypass.

## 6. Test results — pipeline phase (loopback, both peers on one machine)

All 5 pipeline tests × 2 backends = 10 runs, all passing, through the real
`GroupCoordinator.send`/`.recv()`:

| Test | TCP | UDP |
|---|---|---|
| 11: single activation (1/4/16/64 MB, float16) | PASS (64MB: 414ms, 1.3 Gbps) | PASS (64MB: 2.2s, 244 Mbps) |
| 12: 1000 repeated transfers (RTT/jitter/loss) | PASS (avg 0.41ms, 0 lost) | PASS (avg 1.05ms, 0 lost) |
| 13: 512-step simulated decode loop (tok/s equiv.) | PASS (2345 steps/s) | PASS (804 steps/s) |
| 14: bidirectional round trip Stage0→Stage1→Stage0 | PASS (15.8ms, exact) | PASS (65.9ms, exact) |
| 15: 3-stage pipeline Stage0→Stage1→Stage2 | PASS (exact, no corruption) | PASS (exact, no corruption) |

Test 15's TCP "Stage0 send" figure (~2s) is a connection-establishment
timing artifact, not a bug: Stage0's send-timer starts right after its own
`connect()` returns, but `sendall()` can block on kernel send-buffer
backpressure until Stage1 actually starts draining its socket, and Stage1
only starts reading *after* it finishes establishing its *second*
connection (to Stage2) - a real, if test-harness-specific, ordering
dependency of holding two point-to-point links in one process. The UDP run
(no listen/accept handshake) shows all three hops in the same 11-12ms
range, confirming this is a TCP-connection-sequencing artifact of the test,
not a transport or interception-point defect - correctness (byte-exact
propagation through all 3 stages) held in both cases.

## Bugs found this phase

None in the transport or interception-point code. Reusing phase 2's
`TransportProcessGroup` unmodified meant the only new code this phase
added was the 6-line `GroupCoordinator` branch plus the sandbox-import
workaround described above (which is test-only, not a functional fix).

## Honest limitations of this phase

- Only `GroupCoordinator.send`/`.recv()` (single-tensor P2P) was
  intercepted. Collectives (`all_reduce`, `all_gather`, `broadcast`, etc.)
  and the bulk `send_tensor_dict` path used by real multi-tensor
  pipeline-parallel activations are untouched (§4) — this remains a
  proof-of-concept for *one* activation path, as scoped.
- `GroupCoordinator` was never constructed the normal way (with `transport=`
  passed through a real `torch.distributed`-backed `__init__`) — only via
  the `object.__new__` bypass. Wiring `transport_backend` from
  `ParallelConfig` (already present since phase 1) through to an actual
  `GroupCoordinator(..., transport=...)` call at real init time is
  unimplemented and is the next concrete step before this could run
  alongside a real model (still blocked in this sandbox by the same
  torch-version incompatibility noted throughout, independent of this
  phase's own code).
- `TransportProcessGroup` is still point-to-point only (phase 2's
  limitation, inherited here) — test 15's three-stage pipeline works by
  giving the middle stage two independent connections, not by making
  `TransportProcessGroup` itself multi-rank.

---

# Phase 4: `send_tensor_dict`/`recv_tensor_dict` - the real PP call site

Prompted by an attempt to actually deploy GPT-OSS 120B across two
machines (see `README_GPTOSS_120B_UDP.md` at the repo root for the full
deployment doc, gap analysis, and blockers). That effort found that real
`gpu_model_runner.py` pipeline-parallel code calls `get_pp_group().send_tensor_dict()`/
`.recv_tensor_dict()` — **not** the `.send()`/`.recv()` this project
patched in phase 3. Grep evidence: every `get_pp_group().X` call site in
`vllm/v1/` and `vllm/model_executor/` uses `send_tensor_dict`,
`recv_tensor_dict`, `isend_tensor_dict`, `irecv_tensor_dict`,
`broadcast_tensor_dict`, `broadcast_object`, `graph_capture`,
`device_communicator`, or `make_sibling_device_group` - never the raw
`send`/`recv` phase 3 covers (that method genuinely exists and is real -
vLLM's own `tests/distributed/test_comm_ops.py` and
`ray_communicator.py`'s `RayPPCommunicator` use it - just not on this
particular call path).

## What changed

`vllm/distributed/parallel_state.py`:
- Two new module-level helpers, `_transport_send_tensor_dict`/
  `_transport_recv_tensor_dict`, next to the existing `_split_tensor_dict`
  they reuse. Wire format: pickled `metadata_list` (the same
  key/`TensorMetadata`-or-value split the NCCL/gloo path already uses),
  length-prefixed, followed by each tensor's `serialize_tensor()` payload
  in order. CUDA tensors are staged to host memory first (`tensor.cpu()`)
  - `Transport` only ever moves `bytes`, same D2H constraint noted in
  phase 2.
- One `if self.transport is not None: ...` branch each, as the *first*
  line of `send_tensor_dict`/`recv_tensor_dict` - placed before the
  existing `if not torch.distributed.is_initialized() or self.world_size == 1`
  guard those methods already have, since a transport-backed synthetic
  group (phase 3's `object.__new__` construction) has `world_size=2` but
  no real `torch.distributed` backing, so that guard's "just no-op"
  behavior would otherwise silently swallow the send/recv instead of
  routing it.

Tested with `tests/transport/test16_pipeline_tensor_dict.py` (new): a dict
mixing CPU float16 tensors and plain (non-tensor) values, round-tripped
through the real `GroupCoordinator.send_tensor_dict`/`.recv_tensor_dict`
via the same `_pipeline_shim.py` bypass phase 3 introduced. PASS on both
`--transport tcp` and `--transport udp`.

## What this does and doesn't get you closer to

This closes a real, previously-open gap (phase 3 §"Honest limitations"
already flagged that only single-tensor `.send()`/`.recv()` was
intercepted, not the bulk dict path) - real pipeline-parallel activation
dicts can now correctly traverse this transport, verified end-to-end at
the `GroupCoordinator` API level.

It does **not** get GPT-OSS 120B (or any real model) running across two
machines. That gap turned out to be much larger than the transport layer:
vLLM's entire distributed bootstrap (`torch.distributed.init_process_group`'s
TCP rendezvous, and every built-in executor's assumption of direct
reachability between all ranks) has no NAT-traversal-aware path at all
today, independent of how complete the tensor-passing layer is. Full
details, code references, and effort estimates for closing that gap are
in `README_GPTOSS_120B_UDP.md`'s "Remaining blockers" section - it is
substantially larger work than anything done in phases 1-4.

---

# Phase 5: the "impossible" rendezvous problem, solved and proven for real

Phase 4 ended believing `torch.distributed`'s TCP rendezvous was an
unsolvable blocker for two NAT'd machines. It wasn't - the framing was
wrong. `torch.distributed` only ever needs to span the GPUs on *one*
machine (tensor parallelism); it never needs to span both machines at
all. The pipeline-parallel dimension - the one dimension that genuinely
crosses machines - was already being routed through `vllm.transport`
since phase 3-4. This phase's real work was building the piece that
installs that routing into a **live, fully-formed** `GroupCoordinator`
setup instead of only a phase-3-style bare `object.__new__` bypass.

## What changed

- **New**: `vllm/transport/pipeline_bootstrap.py` -
  `install_transport_pp_group(transport, pp_rank, pp_world_size, local_rank)`
  replaces the module-global `_PP` with a synthetic, transport-backed
  `GroupCoordinator`. Call it *after* the real local
  `init_distributed_environment`/`ensure_model_parallel_initialized` has
  already formed the real local TP/DP/EP groups (and a locally-trivial
  `_PP` this function discards) - never before, since that bootstrap's own
  `initialize_model_parallel` asserts `_PP is None` at entry.
- **New**: `tests/transport/test18_real_bootstrap_pp.py` - two processes,
  each forms a REAL local `torch.distributed` group (gloo, loopback,
  genuinely reachable - no NAT involved), then swaps in the transport-
  backed `_PP`, then sends a real `torch.Tensor` through the real,
  unmodified-elsewhere `GroupCoordinator.send_tensor_dict()`/
  `.recv_tensor_dict()`. **PASSES on both `--transport tcp` and
  `--transport udp`.** Also verifies the real local TP group (`all_reduce`)
  still works correctly *after* the PP swap - proving the two coexist
  without interference.
- **Two real, minimal, in-repo compatibility fixes** to `vllm/distributed/parallel_state.py`
  (`list[int]` → `List[int]` in `patched_fused_scaled_matmul_reduce_scatter`,
  blocking `import vllm.distributed` entirely on this sandbox's torch
  version) and `vllm/ir/tolerances.py` (guarded `torch.float4_e2m1fn_x2`
  behind `hasattr`, since that module is imported unconditionally). Both
  are narrow, torch-version-compatibility patches consistent with the
  existing acknowledged-upstream-bug pattern already documented in phase 3
  - not redesigns, not workarounds around this project's own logic.
- **New**: `tests/transport/_env_stubs.py` - consolidates every sandbox-
  only import shim needed to get `vllm.config`/`vllm.distributed`/`vllm`
  (the full engine import chain, not just the transport package) importing
  for real in this environment: `transformers` v5 upgrade, several missing
  pip dependencies, a `tvm_ffi` stub (xgrammar's own missing dependency),
  and forcing CPU platform selection (this checkout has no compiled kernel
  extensions - see below). Every shim documents exactly what it's for and
  what doesn't work because of it.

## What this proves, and what it doesn't

**Proven for real**: the rendezvous/bootstrap problem that looked
architecturally unsolvable in phase 4 is solved. Two independent
processes, each with its own real local distributed group, can have a
transport-backed pipeline-parallel link installed between them, and real
tensor data flows correctly end-to-end through vLLM's actual
`GroupCoordinator` API - not a reimplementation, not a mock.

**Not proven**: real GPT-OSS 120B inference. Getting from here to a real
forward pass hit a different, unrelated wall - this checkout has no
compiled CUDA/CPU kernel extensions (never `pip install`-ed), and
separately, vLLM's model-registry inspection step hits a pervasive
torch-version incompatibility across 60+ files. Both are real, both are
precisely diagnosed with evidence, and both are documented in full - with
proposed solutions, effort estimates, and exactly why this project stopped
short of fixing them itself - in `README_GPTOSS_120B_UDP.md`'s "Remaining
blockers" section. Neither is a transport or distributed-bootstrap
problem; both are pre-existing environment/build gaps this specific
sandbox has, unrelated to anything this project changed.

# Phase 6: a Rust-native `"udp"` backend, replacing the asyncio one

Phases 1-5 built `UDPTransport` on the existing `peer.py` hole-punch
transport, with message framing and reliability (chunk/status-query/
resend) implemented in Python on top of asyncio's `DatagramProtocol` -
one `sendto()`/`recvfrom()` syscall per datagram, dispatched through one
Python callback per datagram. Profiling that architecture (see
`udp_transport.py`'s and `quic_rs_transport.py`'s own module docstrings)
repeatedly found this per-packet Python/asyncio dispatch overhead as the
dominant cost, not the kernel, not the protocol. This phase answers the
question that raised: what's the real ceiling if that architecture is
replaced entirely, and is it worth doing for real traffic?

## Benchmark investigation (loopback, why sendmmsg/recvmmsg batching was chosen)

Built `rust/src/udp_raw_engine/` (Rust, `libc::sendmmsg`/`recvmmsg` -
Linux batched-datagram syscalls, many datagrams per syscall each
direction) as a from-scratch experiment, deliberately *unreliable* at
first, to isolate the real throughput ceiling before paying for
reliability. Rigorous synchronized-timing methodology throughout
(`multiprocessing.Barrier` for both sides' `t0` - an earlier, since-
corrected round of ad-hoc benchmarks understated every number here by
~3.7-4x due to a blind-`sleep()`-based timing bug, documented in full in
the `project_raw_udp_rs_bench` memory entry for anyone re-deriving these
numbers later). 16MB payload, loopback, byte-perfect every run:

| Path | Mbps | Notes |
|---|---|---|
| `udp_transport.py` (asyncio, old) | 263-299 | one syscall/callback per datagram |
| `quic_rs_transport.py` (Rust `quinn-proto`, but Python asyncio I/O loop) | 234-274 | protocol logic in Rust; socket I/O still per-packet Python - see below |
| raw UDP-rs, GSO/GRO batching | 1133-1230 | `UDP_SEGMENT`/`UDP_GRO`, ~65KB aggregate cap per syscall forces more rounds |
| raw UDP-rs, **sendmmsg/recvmmsg batching** | **1672-1813** | chosen path - fewer, larger syscalls per round than GSO's cap allows |
| plain TCP (same rigorous script) | 4204-4634 | reference ceiling |
| raw UDP-rs, multi-socket N=8 (unreliable, loopback only) | 3603-4206 | **not integrated into the reliable engine - see "Explicitly not done" below** |

Two structural findings, not implementation bugs, explain most of the
remaining ~2.3-2.8x gap to TCP: (1) TCP sockets on Linux auto-tune send/
receive buffers up to several MB (`net.ipv4.tcp_rmem`/`tcp_wmem`,
`tcp_moderate_rcvbuf`); UDP sockets are hard-capped at
`net.core.rmem_max`/`wmem_max` (208 KiB on this machine, confirmed not
raisable from inside this container - no `CAP_NET_ADMIN`), with no
per-protocol override; (2) real QUIC packet/protocol overhead
(AEAD encrypt/decrypt, ACK-frame processing, congestion control) is
irreducible and not present in this raw engine at all - see the
`project_udp_vs_tcp_kernel_buffer_policy` and
`project_quic_rs_vs_raw_udp_gap` memory entries for the full evidence
trail on each. Kernel-bypass options (`AF_XDP`, `DPDK`) were checked
directly, not just assumed - both are blocked by this container's
capability set (`ip link ... xdp` and the raw `bpf()` syscall both fail
with real permission errors, `CAP_NET_ADMIN`/`CAP_BPF`/`CAP_SYS_ADMIN`
all absent), so they are not an available lever here.

## What changed (this phase)

| File | What it is |
|---|---|
| `rust/src/udp_raw_engine/src/lib.rs` | `RawUdpEngine` - `send_batch`/`recv_batch` (raw `sendmmsg`/`recvmmsg` primitives), `send_reliable`/`recv_reliable` (single-shot, caller-knows-size - the benchmark pair), `send_message`/`recv_message` (dynamic-size discovery + `msg_id` disambiguation - the pair this backend actually uses), `send_reliable_gso`/`recv_reliable_gro` (GSO/GRO alternative - **still unreliable**, measured slower, not used by this backend) |
| `rust/src/udp_raw_engine/python/src/lib.rs` | `PyRawUdpEngine` PyO3 binding - `from_fd` (dup()s the caller's socket fd), thin wrappers releasing the GIL (`py.detach`) for every blocking call |
| `vllm/transport/udp_rs_raw_bench.py` | `RawUdpBench` - thin benchmark-only wrapper (kept for reference/regression benchmarking, NOT used by `factory.py`) |
| `vllm/transport/udp_rs_transport.py` | **New** - `RawUdpRsTransport`, the real `Transport` implementation this phase adds. Reuses `peer.py`'s hole-punch/STUN unmodified for connection setup; hands the connected socket to `PyRawUdpEngine.from_fd()` for the data path once established; plain-thread keepalive (no asyncio left after handoff) |
| `vllm/transport/factory.py` | `"udp"` now resolves to `RawUdpRsTransport`, replacing `UDPTransport` outright (not added alongside as a separate opt-in name) - `udp_transport.py` itself is untouched and still importable directly if ever needed for comparison |

## Three real bugs found via targeted testing, not code review

Each was found by a DIFFERENT testing technique aimed at a different
failure class - worth remembering as a checklist for future changes to
this engine, not just "add more benchmarks":

1. **Ack-resend deadlock under real packet loss** - found via a custom
   lossy UDP relay proxy (loopback essentially never drops on its own,
   so this needed to be induced). The receiver only re-sent its
   cumulative ack when its VALUE changed since last sent; if that one
   ack packet was itself lost, the receiver believed it had already
   informed the sender and never repeated it, while the sender's own
   resends of already-fully-received data produced no new receiver-side
   state change to prompt a fresh attempt. Fixed: acks now re-send
   unconditionally on a 2ms timer while idle, not gated on "did the
   value change."
2. **~40ms round-trip latency bug** (should be sub-millisecond on
   loopback) - found via a ping-pong latency test. A sender's own
   ack-polling loop could read the peer's already-in-flight reply DATA
   packet in the same `recvmmsg` batch as the ack it was waiting for,
   and silently discarded it (wrong type for that loop), forcing a real
   ~20ms retransmit cycle to recover a packet that had already arrived.
   Fixed: `pending_acks`/`pending_data` stashes on the engine hold onto a
   "wrong type" packet for the OTHER poller instead of dropping it.
   Result: 40ms → ~0.32-0.35ms p50 (500 rounds, 64B messages).
3. **Cross-message data corruption in back-to-back streaming** - found
   via a 300-message randomized-size streaming test (no ping-pong
   turn-taking, sender blasts through messages as fast as each
   `send_message` returns). Message N+1's chunks reused the SAME
   sequence numbers message N used; if the receiver's `recv_message`
   call for message N hadn't returned to Python yet when N+1's early
   chunks arrived (a real, observed race - the kernel buffers them
   regardless), a coincidental sequence-number match spliced message
   N+1's bytes into message N's buffer. Confirmed directly: message
   11/300 came back at a length between two real message sizes. Fixed:
   every chunk now carries a `msg_id`; `recv_message` keeps persistent
   per-`msg_id` receive state across calls (`inbound`/`completed_order`
   on the engine) instead of call-scoped locals.

Full technical detail on all three (exact mechanism, wire-format
changes, code locations) is in the `project_raw_udp_rs_production_readiness`
memory entry.

## Test results (this phase, real hole-punch, both peers on one machine, `--transport udp`)

Existing `tests/transport/test*.py` suite, unmodified, run against the
new backend for real (not just the new engine in isolation):

| Test | Result |
|---|---|
| `test1_hello_world.py` | PASS |
| `test2_payload_sizes.py` (1KB-16MB) | PASS, byte-perfect every size |
| `test3_ping_pong.py` (1000 packets) | PASS, avg RTT 0.518ms, P99 0.846ms, **0.00% loss** |
| `test5_concurrent.py` (4 workers) | PASS, no cross-talk |
| `test6_tensor_small.py` (24 dtype/size cases) | PASS |
| `test7_tensor_large.py` (1-64MB tensors) | PASS, 695-1037 Mbps |
| `test8_tensor_streaming.py` (1000 tensors back-to-back) | PASS, **0 lost, 0 corrupted** - the exact scenario bug #3 above was found in |
| `test9_tensor_bidirectional.py` | PASS |
| `test24_dead_peer_timeout.py` | Not applicable to this backend - this test specifically exercises QUIC's protocol-level idle timeout (`--transport quic`'s own documented purpose); UDP-family transports (old and new alike) never had real dead-peer detection independent of the caller's own `recv()` timeout, so this isn't a regression this phase introduced |

## Explicitly not done this phase (honest gaps)

- **Multi-socket parallelism (N=8, ~3.6-4.2 Gbps unreliable on
  loopback) was investigated but deliberately NOT integrated** into the
  reliable engine. Reconsidered mid-session: that gain is loopback-
  specific (it works around loopback's own local kernel UDP buffer
  ceiling), and on a real cross-machine link the bottleneck is normally
  the physical connection itself, not the local socket buffer - opening
  N parallel flows might not reproduce the gain there, some network
  paths penalize many parallel flows from one host, and it would add N×
  the reliability-state-machine complexity. Left for a future revisit
  IF a real 2-machine benchmark shows it's still needed.
- `send_reliable_gso`/`recv_reliable_gro` did not receive any of the
  three fixes above - still unreliable, still benchmark-only (already
  measured slower than the sendmmsg path, so not a loss).
- NAT-rebind resilience during the DATA phase (the old `UDPTransport`'s
  `_recent_peer_addrs`, sending to every recently-seen address to
  survive one side's NAT round-robining across external IPs) was not
  carried over - `connect()` locks the new engine to a single peer
  address for its whole data-phase lifetime. Hole-punch itself still
  detects/logs a rebind during connection setup.
- No dead-peer/idle-timeout detection independent of the caller's own
  `recv(timeout=...)` - same gap the old `UDPTransport` had; only the
  QUIC backends solve this structurally via the protocol's own idle
  timeout.

# Phase 7: QUIC gets the same treatment - one Rust-native `"quic"` backend

Phase 6 moved raw UDP's whole data path into Rust. This project had TWO
QUIC backends at the time: the original aioquic-based `QUICTransport`
("quic") and a partial Rust port ("quic-rs") that moved only the protocol
*state machine* (`quinn-proto` via `Engine`) to Rust while socket I/O,
timer scheduling, and event draining still ran through Python asyncio -
see [[project_quic_rs_vs_raw_udp_gap]] (memory) for why that architecture
left "quic-rs" slower than raw-udp-rs despite the protocol logic already
being native. This phase finishes the job: the ENTIRE connection
lifetime - handshake, timers, GSO send, stream framing, drain-before-
close - now runs on a dedicated Rust thread, and the two old backends
were deleted outright and consolidated into one `"quic"` name (not kept
alongside as a third opt-in variant) per explicit instruction.

## What changed

| File | What it is |
|---|---|
| `rust/src/quic_engine/src/driver.rs` | **New.** `ConnectionDriver` - owns a real `UdpSocket` + the existing sans-io `Engine`, drives the whole connection on a background thread. Cross-thread comms via `std::sync::mpsc` channels (`send()`/`recv()`/`connect()` each block on a per-call or per-connection channel, GIL released by the PyO3 wrapper) |
| `rust/src/quic_engine/python/src/lib.rs` | Added `PyQuicConnectionDriver` (`connect_client`/`connect_server`/`send`/`recv`/`close`) alongside the existing lower-level `PyQuicEngine` (kept, unused by any transport now, same "keep the lower-level primitive around" precedent `udp_raw_engine` already set) |
| `vllm/transport/quic_transport.py` | **Rewritten in place** - old aioquic content replaced with `QUICTransport` built on `PyQuicConnectionDriver`. Hole-punch reused unmodified from `peer.py`, same handoff pattern `udp_rs_transport.py` established (asyncio drives ONLY hole-punch, then torn down entirely) |
| `vllm/transport/quic_rs_transport.py` | **Deleted** - the old asyncio-orchestrated Rust-backed transport, fully superseded |
| `vllm/transport/factory.py` | `"quic-rs"` branch removed entirely; `"quic"` now resolves to the rewritten `quic_transport.py`. `"quic-shared"`/`"quic-rs-shared"` (multiplexed channels, `quic_broker.py`/`quic_rs_broker.py`) explicitly confirmed out of scope and left untouched - different capability, not another way to do the same thing |
| `vllm/transport/base.py` | Removed 3 now-dead `TransportConfig` fields (`quic_congestion_control_algorithm`/`quic_max_data`/`quic_max_stream_data`) - aioquic-only tuning knobs nothing reads anymore now that backend is gone |

One real architectural simplification fell out of the port, not just a
straight translation: the old design's 1-byte `_QUIC_TAG` wire prefix
existed only because asyncio's single dispatcher shared one socket
between hole-punch control traffic and QUIC application data. Since
`ConnectionDriver` only ever takes ownership of the socket AFTER
hole-punch fully completes (same handoff pattern as the raw UDP
backend), nothing else ever reads the socket again - the tag byte is
unneeded in the new design, not merely carried over.

## A real bug found via testing methodology, not code review

A single 16MB `send()` hung forever; an isolated 300-message streaming
test (up to 500KB each) passed on the very first try, never exercising
the buggy path. Root cause: the edge-triggered "wait for a
`StreamEvent::Writable` event before retrying a blocked stream write"
gate was implemented as `pending.offset > 0` (true for ANY message with
some data already written) instead of "was the immediately-preceding
write attempt genuinely blocked." Since `Writable` only fires on a
transition INTO blocked and back OUT (a real, documented quinn-proto
contract - `WriteError::Blocked` folds into `Ok(0)`), a message spanning
multiple driver-loop iterations without ever actually exhausting the
flow-control window would wait forever for an edge that could never
occur. This is the EXACT deadlock class `quic_rs_transport.py`'s own
`send_message` docstring already documented fixing once for the Python
version (large single messages used to stall completely, per that file's
"KNOWN, UNRESOLVED LIMITATION... RESOLVED" history) - reintroduced fresh
in this new Rust port via a subtly wrong gate condition, caught only
because a large-message test was run separately from the
already-passing small-message streaming test. Fixed with a dedicated
`blocked_on_writable` flag set only on a genuine `Ok(0)` result.

## Test results (real hole-punch, both peers on one machine, `--transport quic`)

Full existing suite, unmodified, all PASS:

| Test | Result |
|---|---|
| `test1_hello_world.py` | PASS |
| `test2_payload_sizes.py` (1KB-16MB) | PASS, byte-perfect every size |
| `test3_ping_pong.py` (1000 packets) | PASS, avg RTT 0.234ms, **0.00% loss** |
| `test5_concurrent.py` (4 workers) | PASS, no cross-talk |
| `test6_tensor_small.py` (24 dtype/size cases) | PASS |
| `test7_tensor_large.py` (1-64MB) | PASS, 346-508 Mbps |
| `test8_tensor_streaming.py` (1000 tensors) | PASS, 0 lost/corrupted, 337 Mbps |
| `test9_tensor_bidirectional.py` | PASS |
| `test22_tensor_500mb_memory.py` | PASS - 500MB single transfer (8.67s) + 50x10MB repeat, memory flat (the exact scenario the old aioquic docstring documented as historically fragile) |
| `test23_fault_injection_lossy.py` (5% drop/3% dup/15% reorder) | PASS, byte-perfect despite injected loss |
| `test24_dead_peer_timeout.py` (real SIGKILL) | **PASS** - detected in exactly 5.00s matching the configured `idle_timeout`, something the `udp` backend structurally cannot do (real protocol-level connection state, not just an application-level `recv()` timeout) |

Isolated (no hole-punch, direct `PyQuicConnectionDriver` use) rigorous
`multiprocessing.Barrier`-synchronized throughput: 694-741 Mbps for a
single 16MB message - compared to the old Python-orchestrated "quic-rs"
(234-274 Mbps, same methodology): **~2.7-3x faster**.

## Explicitly not done this phase (honest gaps)

- `simulate_rebind`/`test21_connection_migration.py` not run against the
  new driver - no equivalent connection-migration test hook built yet.
- Real QUIC-level keepalive PING still not exposed by `ConnectionDriver`
  (same gap the old Python-orchestrated version had) - keepalive stays
  NAT-pinhole-only via `peer.py`'s own ping tag.
- `quic_rs_broker.py` (kept, backs `"quic-rs-shared"`) still has several
  comments referencing the now-deleted `quic_rs_transport.py` by name -
  harmless prose staleness, deliberately left alone since that file was
  explicitly confirmed out of scope.
- ~24 `tests/transport/test*.py` files' `argparse` `--transport` choices
  lists still list `"quic-rs"` - passing it now raises a `ValueError`
  from `factory.py` rather than being rejected at CLI-parse time.
  Cosmetic, not fixed this phase.

# Phase 8: `"quic-shared"` gets the same treatment - one Rust-native broker

Same consolidation as phase 7, applied to the multiplexed-channel broker
(several logical channels - per-TP-rank PP links plus the RPC control
channel - sharing ONE real QUIC connection instead of each hole-punching
its own). `MultiplexedConnectionDriver` (new,
`rust/src/quic_engine/src/multiplexed_driver.rs`) is the multi-channel
analogue of phase 7's `ConnectionDriver`, sharing its GSO/cmsg helpers.
`"quic-shared"`/`"quic-rs-shared"` consolidated into one `"quic-shared"`
name per explicit confirmation - old aioquic `quic_broker.py` and the old
Python-orchestrated `quic_rs_broker.py` (+ both daemon launcher scripts)
deleted outright. `quic_broker_common.py` (new) holds the genuinely
backend-agnostic local-Unix-socket IPC plumbing, extracted so it's no
longer bundled inside the aioquic-specific file. `quic_multiplexed_transport.py`
(the actual client-facing `Transport`) needed zero changes - already
100% backend-agnostic.

Also fixed real staleness found while consolidating: `scripts/stage_server.py`/
`scripts/launch_pp_stage.py`'s daemon-launcher logic, several
`--transport` `choices` lists, and two membership checks still treated
`"quic-rs"`/`"quic-rs-shared"` as live options (some missed during phase
7 itself - a reminder to grep the whole repo for a retired name, not
just the transport package).

## A severe production bug: `close()` could hang a real daemon forever

Found by testing the REAL `quic_broker_daemon.py` subprocess end-to-end
(launch, connect, SIGTERM, expect clean exit) - a bare in-process
`connect()`-then-`close()` test passed fine, but the real daemon (which
has a background dispatch loop continuously blocked in a `recv`-style
call) hung indefinitely. Root cause, confirmed by testing `close()`
directly while a separate thread had a blocking `recv` call genuinely in
flight: **a PyO3 dynamic borrow-checking conflict, not a Rust logic
bug**. `close()`'s PyO3 wrapper was `&mut self` (needs an EXCLUSIVE
borrow); `recv()`/`recv_any()` (`&self`) held a SHARED borrow for its
entire blocking duration - `py.detach()` releases the Python GIL during
that call but NOT PyO3's own borrow guard. Rust can never grant an
exclusive borrow while a shared one is outstanding, so `close()` raised
`RuntimeError: Already borrowed` before its body ever ran - `shutdown`
was never set, hanging the thread (and the whole process) forever. The
exception was silently swallowed by a `contextlib.suppress(Exception)`
already present in the Python wrapper, making this fail completely
silently in production - no error, no log, just a daemon that never
exits. **Affected BOTH drivers** (`"quic"` and `"quic-shared"`), not just
the new multiplexed one - see [[project_quic_shared_rust_native_broker]]
(memory) for the full account, including a RETROACTIVE correction added
to phase 7's own memory entry.

Fixed by changing `close()` to `&self` at both the PyO3 wrapper level and
the underlying Rust struct level, using `std::sync::Mutex<Option<JoinHandle<()>>>`
(interior mutability) instead of a plain `Option<...>` field for the one
genuinely-mutating operation. Confirmed fixed: a direct repro (thread
blocked in `recv_any()`, `close()` called concurrently) now completes in
~10ms; the real daemon subprocess now shuts down in ~0.8s instead of
hanging forever.

**A second, related bug** found in the same investigation, before even
finding the borrow conflict: `close()` waited for a `drained` flag
BEFORE setting `shutdown` - but the driver thread only sets `drained`
AFTER observing `shutdown`. A pure wait-before-signal deadlock that made
EVERY `close()` call take exactly the full `drain_timeout`, unconditionally,
even with nothing to drain (confirmed: a bare connect-then-close took
exactly 3.01s, matching `drain_timeout` to the millisecond). The grace
period the thread actually used internally was also hardcoded to 200ms,
ignoring whatever `drain_timeout` the caller passed. Fixed by setting
`shutdown` first and threading the real timeout through a shared atomic.
Confirmed fixed: isolated `close()` with nothing to drain now takes
0.1-0.4ms, not 3000ms - and `test5_concurrent.py`'s 4-worker wall time
dropped from 9.20s to 6.17s as a direct, measurable consequence.

## Test results

`test26_quic_broker_multiplexing.py` (3 channels, 20 messages each
direction, real local-socket IPC hop): PASS - order preserved, zero
cross-channel leakage, single shared connection confirmed (the test
itself needed a small fix too - it inspected `broker._quic`, an
aioquic-only attribute that no longer exists; updated to check
`broker._driver`'s identity instead, preserving the test's actual
intent). Real daemon end-to-end (subprocess launch, `get_transport
("quic-shared")` + `TransportConfig.extra`, real message exchange, clean
SIGTERM shutdown): PASS. Existing `"quic"` single-channel suite re-run
after the shared `close()` fix: still PASS, same numbers as phase 7
within noise.
