# transport_runtime

A framework-agnostic distributed communication runtime. This is the
**Phase 2A** extraction described in `docs/ARCHITECTURE_DECISION.md`
(in the sibling `vllm/` checkout) — see that document for the full decision
history (why extract at all, why vLLM stays as the first adapter, what was
debated and revised across two rounds of architecture review).

## Relationship to `vllm/transport/`

`vllm/transport/{base,factory,tcp_transport,udp_transport,tensor}.py` in
the sibling checkout are left in place, unmodified — `tests/transport/
test1`-`test9` (transport-layer-only tests) still import them directly.
**Phase 3 is done**: `vllm/transport/pipeline_bootstrap.py` and
`vllm/distributed/parallel_state.py`'s transport-backed send/recv paths
now consume this package instead of those vendored files. This was
verified regression-free on real hardware in this project's actual
environment (2× Tesla T4) — `test18`/`test20` (real cross-process PP
bootstrap + tensor exchange, TCP and UDP) and `test11`-`test16` (the
`GroupCoordinator.send()`/`.recv()` single-tensor path this migration
touched) all re-passed after the migration, not just byte-compiled.

One important correction from attempting this migration for real: the
original plan (see `docs/ARCHITECTURE_DECISION.md` Part 6, first
draft) was to also replace `pipeline_bootstrap.py`'s `object.__new__`
singleton-mutation pattern with a `CommunicatorBase`-conformant adapter
built through `GroupCoordinator`'s real constructor. That turned out to
be infeasible without a much larger project: `GroupCoordinator.__init__`
unconditionally calls `torch.distributed.new_group()`, which requires an
already-rendezvoused default process group and does not accept a custom
`Store` — so there is no way to reach the "sanctioned" pattern for a
cross-NAT PP group without first building a full `transport_runtime`
-backed `torch.distributed.Store`. `object.__new__` singleton mutation is
therefore understood to be structurally necessary here, not a shortcut
avoiding a cheaper known-better alternative. See
`docs/ARCHITECTURE_DECISION.md` Part 7 Phase 3 for the full
finding.

## What changed relative to the vLLM-fork version, and why

Both changes close gaps found during architecture review, before any of
this was implemented (see `docs/ARCHITECTURE_DECISION.md`'s final
review round):

1. **`TransportConfig` → `ConnectParams` + `TCPBackendConfig`/
   `UDPBackendConfig`.** The original was one flat dataclass with both
   TCP fields and UDP fields, each backend silently ignoring the other's.
   That made the "swap the backend, call sites don't change" claim true
   of call sites but not of the config's own shape. Now `TCPBackend` only
   ever reads `params.tcp` and physically cannot see `params.udp`.
2. **An explicit error/liveness contract.** The original interface never
   said whether `recv()` could distinguish "nothing has arrived yet" from
   "the peer is gone and nothing ever will" — both looked like the same
   generic hang/timeout. `ConnectionClosedError` (see `backend.py`) now
   makes that distinction explicit, with an honestly-documented asymmetry
   between TCP (can detect real peer-side close) and UDP (best-effort;
   only a *local* `close()` is guaranteed to surface as this error).

## Layering

```
Backend            TCPBackend / UDPBackend — moves bytes between two named
                    endpoints. Knows nothing above this line.

Codec               TensorCodec / JSONCodec / BytesCodec — turns real
                    objects into bytes and back. Knows nothing about
                    Backends or Connections.

Connection /        one point-to-point link (Backend + Codec) / a flat
ConnectionManager   peer_id -> Connection registry. NOT a topology graph —
                    see Non-Goals below.

(not in this        FrameworkAdapter — vLLM/SGLang/etc-specific glue.
 package)           Phase 3+, lives in each framework's own checkout/adapter
                    package, imports this package, this package never
                    imports it.
```

## Non-Goals

(Carried over verbatim in spirit from `docs/ARCHITECTURE_DECISION.md`
— restated here because this is the package those goals actually bind.)

- Not a cluster scheduler or orchestrator. Moves bytes between named
  endpoints; does not decide what work goes where.
- Never owns scheduling or placement policy — only communication
  mechanism.
- Does not own network topology as a modeled graph. `ConnectionManager`
  is a flat `peer_id -> Connection` registry; it has no `Node`, no
  `Edge`, no concept of "this is a pipeline." Whoever calls `connect()`
  repeatedly is the one who knows the shape of the whole.
- Does not know or care whether it's serving training or inference, and
  does not interpret payload semantics — a `Codec` turns objects into
  bytes; what those objects *mean* (token, activation, gradient, KV
  cache, ...) stops at that boundary.
- Not an inference engine, not a quantization framework.
- Does not reimplement congestion control, multipath, or QoS by hand.
  If needed later, they come from adopting a backend that already solves
  them (e.g. a future QUIC backend via `register_backend()`), not from
  hand-rolled protocol work here.
- Does not implement adaptive/automatic backend-selection policy (e.g.
  auto-switching UDP→TCP based on measured loss). Backend choice is an
  explicit caller decision, not something this package decides on its
  own.

## Known limitations (stated, not hidden)

- `Codec.encode()/decode()` is `bytes`-in/`bytes`-out. For CPU tensors
  (`TensorCodec`, the only proven payload type so far) that means a real
  memory copy per call — not zero-copy, no pinned memory, no GPU
  staging. Deferred to Phase 6 (performance tuning) if profiling actually
  shows it matters; not solved speculatively here.
- `UDPBackend` can only guarantee `ConnectionClosedError` for a *local*
  `close()`. A UDP peer that silently vanishes surfaces as an ordinary
  `TimeoutError` — UDP has no FIN, and this package does not paper over
  that with a false liveness guarantee.
- **Resolved by Phase 2B** (`examples/synthetic_dispatch/`): the adapter
  boundary is **Communication** (send/recv *and* scheduling-step
  dispatch — both are just repeated `Connection.send()`/`.recv()`) vs.
  **Lifecycle** (bootstrap only, one-time). The first-draft "Executor/
  Lifecycle Adapter" grouping (dispatch + bootstrap together) was wrong;
  see `docs/ARCHITECTURE_DECISION.md` Part 5 for the corrected diagram
  and the exercise that justified it.

## Testing

- `tests/test_swap.py` — the swap test: the *same* call-site code runs
  an echo exchange once with `backend_name="tcp"`. This is the Phase 2A
  exit criterion from `docs/ARCHITECTURE_DECISION.md` — a UDP variant
  of the same test is included but skipped automatically if the external
  hole-punch library isn't importable (it lives outside this package; see
  `backends/udp.py`), rather than failing the suite in an environment
  that hasn't set it up.
- `tests/test_codec.py` — `TensorCodec` round-trip correctness across a
  handful of dtypes/shapes.
- `tests/test_liveness.py` — failure-injection: verifies
  `ConnectionClosedError` actually fires when a peer closes, rather than
  `recv()` hanging forever.
- `examples/plain_pytorch_pipeline/` — Phase 2B litmus test: a real
  3-stage pipeline over 3 subprocesses, zero framework vocabulary
  (enforced by an automated grep, not just manual review), output
  verified against a non-pipelined ground truth.
- `examples/synthetic_dispatch/` — Phase 2B adapter-boundary exercise:
  proved scheduling-step dispatch is a Communication concern, not a
  Lifecycle one (see Known Limitations above).

Run with:

```bash
pip install -e ".[test]"
pytest tests/ examples/ -v
```
