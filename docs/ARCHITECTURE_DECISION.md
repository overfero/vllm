# Architecture Decision: vLLM Fork vs. Standalone Transport Runtime

**Status**: decision document only — no code changed as part of this document.
**Date**: 2026-08-02. Parts 5–7 were refined through two rounds of adversarial
review against a second AI agent after the initial draft; the version below
is the converged design, not the first draft.
**Scope**: whether to continue extending the current vLLM fork for cross-NAT
pipeline-parallel GPT-OSS-120B serving, or extract the transport layer into a
framework-agnostic runtime.

**Path note**: paths below (e.g. `/kaggle/working/transport_runtime/`)
reflect this repo's layout *at the time this decision was made*, before
`udp_holepunch/`, `transport_runtime/`, `humming_fix/`, `ops/`, `pp_tests/`
were consolidated as subdirectories of this checkout. The conclusion (the
3-layer split) is unaffected; see the top-level `README.md` for the
current layout.

This document was produced by (1) direct inspection of this repository —
file contents, `git` history, line counts, and every integration point
between `vllm/transport/` and vLLM core — and (2) live research into the
current (2026) state of vLLM, SGLang, TensorRT-LLM, llama.cpp,
DeepSpeed-Inference, and HuggingFace TGI. Every claim below is labeled
**[verified]** (checked directly against source/live data in this session)
or **[inference]**/**[speculative]** (reasoned conclusion or unverified
claim). Where a prior project document made a claim, it's cited by
filename rather than re-derived.

**Vision statement**: this project is not "transport for vLLM." It is a
**distributed communication runtime, with a vLLM adapter as its first
implementation.** vLLM, SGLang, and any future framework are customers of
the runtime, not its reason for existing.

## Non-Goals

Stated up front because a good architecture document is as clear about what
it will never do as what it will do — this list exists to give future PRs a
one-question test: *"is this mechanism or policy?"* If policy, it doesn't
belong here.

- The runtime is **not** a cluster scheduler or orchestrator (not Ray, not
  Kubernetes). It moves bytes between named endpoints; it does not decide
  what work goes where.
- The runtime **never owns scheduling or placement policy** — only
  communication mechanism. It does not choose which GPU/node/rank runs what.
- The runtime does not own network **topology** as a modeled graph (no
  `Node`, no `Edge`, no "this is a pipeline" semantics). It owns
  point-to-point `Connection`s only. Whoever calls `connect(peer_id)` is the
  one who knows the shape of the whole — the runtime never does.
- The runtime does not know or care whether it is serving **training or
  inference**, and does not interpret payload semantics (token, activation,
  gradient, optimizer state, KV cache are all just bytes to it). That stops
  at the Codec/Adapter boundary.
- The runtime is not an inference engine and not a quantization framework —
  it has no opinion on model formats, kernels, or numerics.
- The runtime does not reimplement congestion control, multipath, or QoS by
  hand on top of raw UDP. If those are needed, they come from adopting a
  backend that already solves them (e.g. QUIC), not from hand-rolled
  protocol work inside this project.

---

## Part 1 — Maintainability of the current vLLM fork

### 1.1 What "the fork" actually is, precisely

**[verified]** `/kaggle/working/vllm` is a real git repo, but its history is
not usable as a diff base: it contains two commits, the second of which
(`66343095e`, *"Reconstructed transport project + this session's
PP-integration work"*) is a single squashed commit that added **6,395 files
/ 1,847,861 insertions** — i.e., the entire vLLM tree was committed as one
blob with no recorded parent revision from upstream. There is no tag, no
`VLLM_VERSION_OVERRIDE`, no commit hash anywhere in the tree
(`vllm/_version.py` hardcodes `__version__ = '0.1.0'`,
`__commit_id__ = None`). **This means the fork currently has no way to
compute "how far have we diverged from upstream" — that number is
unknowable as the repo stands today.** This is itself a maintainability
finding, independent of anything else in this document: whoever inherits
this project cannot `git fetch upstream && git diff` to see what changed.

**[verified]** The actual custom code is small and concentrated. Two
categories:

1. **The transport package** (`vllm/transport/`, 1,022 LOC across 9 files:
   `base.py`, `factory.py`, `tcp_transport.py`, `udp_transport.py`,
   `tensor.py`, `pipeline_bootstrap.py`, `pp_worker.py`, plus `__init__.py`
   and a design-note `README.md`) — this is genuinely self-contained.
   Everything except `pipeline_bootstrap.py` (233 LOC) and `pp_worker.py`
   (132 LOC) has **zero vLLM imports**. That's 657 of 1,022 lines
   (64%) that don't know vLLM exists at all.
2. **Integration into vLLM core**, concentrated in
   `vllm/distributed/parallel_state.py` (~15 discrete touch points: new
   `transport`/`transport_prev`/`transport_next` attributes on
   `GroupCoordinator`, and transport branches added to `send`, `recv`,
   `send_tensor_dict`, `recv_tensor_dict`, `isend_tensor_dict`,
   `irecv_tensor_dict`), plus small plumbing in `vllm/envs.py`,
   `vllm/config/parallel.py`, and `vllm/engine/arg_utils.py`. Notably,
   `vllm/v1/worker/gpu_worker.py` itself has **zero** transport-related
   changes — the worker override happens entirely via `TransportPPWorker`
   subclassing and `--worker-cls`, a sanctioned vLLM extension point.
3. Plus five small, real bugfixes for GPT-OSS/GPTQ/Humming correctness
   (`moe_wna16.py` +40, `gpt_oss.py` +115/-24, `int_wna16.py` +6,
   `pipeline_bootstrap.py` +10) — these are upstream-quality correctness
   fixes, not hacks, and are unrelated to the transport architecture
   question.

So the honest coupling surface is **small in line count** (~15 touch
points, well under 200 lines in vLLM core itself) but **structurally
deep**: it replaces a module-level singleton (`ps._PP`) with an object
built via `object.__new__(GroupCoordinator)` that deliberately bypasses
`__init__`, and it depends on knowing, by trial and error, which
attributes real code paths read off a `GroupCoordinator` at times other
than construction. That's exactly what Bug 5 was
(`README_LIVE_DEPLOYMENT_LOG.md`): `prepare_communication_buffer_for_model`
unconditionally reads `.device_communicator`/`.mq_broadcaster`, a contract
nowhere documented, only discovered by hitting a real `AttributeError` in
a code path (`Worker.load_model()`) that the existing tests
(`test18`/`test20`) never exercise, because both bypass real model loading
entirely (`world_size=1` loopback groups, no `EngineArgs`, no real
`Worker`). **[inference]**: this pattern — "guess the undocumented
attribute contract, get an AttributeError from a real code path our tests
don't cover, patch it" — is not a one-time cost. It's the recurring
failure mode of monkey-patching a private singleton in a codebase that
doesn't know a synthetic object is standing in for a real one. Every new
vLLM code path that reads a new `GroupCoordinator` attribute for the first
time is a candidate for a repeat of Bug 5.

### 1.2 How fast vLLM's internals actually move

**[verified, live research]**:
- vLLM ships on a **two-week release cadence** with hundreds of commits
  and 300+ contributors per release as of mid-2026.
- The distributed/comm layer is **not** settled, legacy code being
  maintained — it already underwent one full rewrite (V0→V1, scheduler +
  executor + worker) and is currently the subject of an **open RFC to
  route collectives through PyTorch's new `torchcomms` layer**, i.e. the
  exact module this fork hooks (`GroupCoordinator`) is actively being
  generalized again right now, not frozen.
- Countervailing evidence, **corrected after Phase 3's actual
  implementation attempt** (see Part 7 Phase 3 for the full finding): vLLM
  does maintain an intentional `CommunicatorBase` abstraction in
  `vllm/distributed/device_communicators/`, and its KV-cache
  disaggregation path (`NixlConnector`) is a real, production, pluggable
  transport-backend precedent (RDMA/UCX, TCP fallback, custom
  out-of-tree backends) — so vLLM *does* have a sanctioned pluggable-
  transport pattern **for communicators used inside an already-
  rendezvoused process group**. It is **not**, on inspection of the real
  `GroupCoordinator.__init__` (which unconditionally calls
  `torch.distributed.new_group()`, itself requiring an already-
  rendezvoused default group `new_group()` cannot bootstrap on its own),
  a usable replacement for what `object.__new__` singleton mutation does
  here specifically — building a PP group that *is* the cross-NAT
  rendezvous, before one otherwise exists. That distinction was not
  verified before the first draft of this document; it has been since.
  The `object.__new__` pattern is structurally necessary for this
  project's actual problem, not merely an available-but-unused
  convenience.

### 1.3 Humming (third-party quantization kernel)

**[corrected in Phase 5 — see below]**: `humming` is pip-installed,
NVRTC-JIT compiled, and **not closed-source** — its Python orchestration
layer and CUDA kernel templates (`.cuh`, compiled from plain-text source
at runtime) are both fully readable, contrary to what was assumed here
originally. It had two confirmed, reproducible bugs on T4/SM75 for
GPT-OSS's exact shapes (a tiling bug and an MMA-instruction-shape bug);
both are now root-caused and fixed via runtime monkeypatch, verified on
real hardware (`humming_fix/README.md`, Part 7 Phase 5). It is not
vendored — patches live in this project's own `humming_fix/` package, not
in the installed package — and has no known release cadence or
maintenance commitment visible from here — its long-term availability is
still an **unknown, not zero risk**, but this risk is orthogonal to the
vLLM-vs-standalone-runtime question: it would exist identically if this
project used SGLang with the same GPTQ checkpoint and the same Humming
backend (confirmed in Phase 4: SGLang depends on
`humming-kernels[cu13]==0.1.10` directly), or
disappear identically if the project switches to a different quantization
path on either framework. **Do not let the Humming blocker itself drive
the fork-vs-extract decision — it doesn't bear on it.**

### 1.4 The gap this audit surfaced that the user didn't ask about but should know

**[verified, from README_RUN_GPTOSS_CLUSTER.md, re-confirmed this
session]**: even setting Humming aside, real multi-machine *serving*
(not just the tensor-level PP send/recv proven by test18/test20) is not
built. `MultiprocExecutor` (`vllm/v1/executor/multiproc_executor.py:354`)
dispatches `scheduler_output` to workers via `MessageQueue`/TCP **within
one machine only** — nothing today propagates a scheduling step across
machines. Closing this gap was already scoped in project docs at
**~400–700 additional LOC** in a new `vllm/transport/rpc_executor.py` +
`stage_server.py`. This is relevant to Part 1 because it's *additional*
future coupling to `vllm/v1/executor/` — a part of vLLM that changed
completely in the V0→V1 rewrite and has no reason to be considered more
stable than `parallel_state.py`. Continuing on the current path means
this coupling surface grows, not shrinks.

### 1.5 Maintenance burden estimate, 1–2 years

| Risk | Likelihood | Impact if it hits |
|---|---|---|
| `GroupCoordinator`/`torchcomms` RFC lands, changes attribute contract or `__init__` shape | Medium-high (already an open RFC) | Breaks the `object.__new__` bypass silently; another Bug-5-style hunt |
| New code path reads a new undocumented `GroupCoordinator` attribute | High (already happened once, unit tests don't cover model-loading) | Same class of bug, one at a time, indefinitely |
| `MultiprocExecutor`/scheduler internals change (V1 already rewrote this once) | Medium | Breaks the not-yet-built cross-machine RPC executor before it's even finished |
| Humming abandoned or incompatible with a future vLLM/torch pin | Unknown, non-trivial | Forces a quantization-backend change regardless of fork/extract decision |
| Fork falls further behind an untracked upstream base | Certain, compounding | Security/perf fixes must be manually re-applied; eventually a full re-fork |

**Bottom line for Part 1**: the current coupling is *small* but sits on
the single most actively-churning seam in vLLM's codebase, uses an
unsanctioned integration pattern (bypassing `__init__`, mutating a
private module global) where a sanctioned one (`CommunicatorBase`)
already exists, and the untracked fork history means every future vLLM
upgrade is a from-scratch reconciliation, not a `rebase`. This is a real,
evidenced, non-speculative maintenance risk — not a hypothetical one.

---

## Part 2 — Continuing vLLM vs. an independent transport runtime

### Continuing to patch vLLM directly

**Advantages**:
- Zero abstraction tax — the transport talks straight to the real
  `GroupCoordinator` vLLM code actually uses at runtime, no adapter
  translation layer to get wrong.
- All the hard-won, real bug fixes (5 of them, GPU-verified) apply
  directly, today, with no re-porting.
- vLLM is (per Part 3) the most likely of all six frameworks to still be
  the dominant open-source serving engine in 2 years — betting on it
  specifically is a reasonably safe bet, just not a *cheap* one to
  maintain.

**Disadvantages**:
- Every future vLLM internal change is a maintenance event on code this
  project doesn't own (§1.5).
- The integration pattern used (private singleton mutation) is exactly
  the kind of "heavily patched fork" the user explicitly said they don't
  want to be locked into — and it's avoidable, since vLLM already
  exposes a sanctioned alternative (`CommunicatorBase`) that was not
  used.
- No portability: the ~365 vLLM-facing lines of `vllm/transport/` provide
  zero reuse value if the project ever wants SGLance/llama.cpp/anything
  else — they'd need to be rewritten, not extracted.
- The untracked fork history compounds every additional month.

### Extracting transport into an independent runtime, vLLM as first adapter

**Advantages**:
- 64% of the existing transport code (`base.py`, `factory.py`,
  `tcp_transport.py`, `udp_transport.py`, `tensor.py` — 657 LOC) already
  has zero vLLM imports. Extraction is not a rewrite, it's a **move** —
  this is unusually cheap for what's normally a large refactor.
- Isolates the volatile seam: when vLLM's comm layer changes, only the
  adapter (~365 LOC) needs updating, not the transport, the hole-punch
  protocol, the signaling server, or any future SGLang/other adapter.
- Makes the SGLang finding in Part 4 — that SGLang's `GroupCoordinator`
  is structurally near-identical to vLLM's (same `send`/`recv`
  if/else shape, even a `monkey_patch_vllm_parallel_state()` interop
  shim) — directly actionable: the adapter pattern built for vLLM ports
  with modest effort, not a rewrite.
- Directly answers the user's stated goal ("I do not want to become
  permanently locked to a heavily patched fork of vLLM").

**Disadvantages**:
- A generic `TransportRuntime` interface is itself new code and a new
  design surface to get wrong — if the abstraction doesn't match what a
  second framework actually needs (see SGLang's `P2PWork`
  async-completion contract, which vLLM's `GroupCoordinator` doesn't
  have an equivalent of), it leaks, and you're maintaining an
  abstraction *and* two leaky adapters instead of one direct
  integration.
- Framework-specific orchestration work does not go away. The
  `MultiprocExecutor` cross-machine scheduling gap (§1.4) is a vLLM
  executor-layer problem; SGLang's chunked-PP rework and `P2PWork`
  wrapper requirement (§4) is a *different*, SGLang-specific
  orchestration problem. A standalone transport runtime removes
  duplicate low-level plumbing, not the framework-specific glue above
  it.
- For a single-operator research project (not a team maintaining
  multiple production adapters concurrently), building for
  multi-framework reuse before a second framework is actually needed is
  a real premature-abstraction risk — mitigated here only because the
  extraction cost is unusually low (see above) and the user has
  explicitly asked for it.

---

## Part 3 — Inference ecosystem comparison

*(Full per-framework research, live-sourced 2026-08-02, is in the
research agent's report reproduced in the appendix section at the bottom
of this document. Table below is the actionable summary.)*

| Framework | Popularity/Activity | PP comm primitive | Transport-swap feasibility | Maintenance burden if forked | Future compatibility |
|---|---|---|---|---|---|
| **vLLM** | 88k★, ~3,100 contributors, 2-week releases, PyTorch Foundation | `GroupCoordinator`: pynccl → `torch.distributed` fallback | High for communicators inside an already-rendezvoused group (`CommunicatorBase`/`NixlConnector`); **not usable for the cross-NAT bootstrap problem itself** — confirmed by Phase 3, see Part 7 | Medium — narrow seam, fast-moving upstream | High — de facto reference implementation, NVIDIA NIM 2.0 itself standardized on it |
| **SGLang** | 31k★, ~1,700 contributors, 400k+ production GPUs, PyTorch ecosystem project | Near-identical `GroupCoordinator` fork of vLLM's, plus `mooncake`/`naive_distributed` backend precedents | High-medium — same seam as vLLM, plus a `P2PWork` async-completion contract to satisfy in `scheduler_pp_mixin.py` | Medium-high — PP actively being rewritten (chunked PP, Jan 2026) | High-medium — huge momentum, smaller core team than vLLM |
| **TensorRT-LLM** | 14k★, small OSS contributor base, NVIDIA-driven | NCCL compiled into the TensorRT engine graph at build time | **Low** — not a runtime-swappable Python object, requires patching a closed compiled engine | High — C++/CUDA, closed format | Medium — NVIDIA's own NIM 2.0 moved to vLLM as default, signaling deprioritization |
| **llama.cpp** | 122k★, 1,000+ contributors, continuous releases | `ggml_backend_rpc`: sequential remote-op proxying over custom TCP, **no NCCL at all** | Medium-high for transport, but conceptually mismatched — no overlapped PP to plug into, would need building PP from scratch on top of it | Medium — small stable core, fast-moving upstream | High for edge/local; not a datacenter PP competitor |
| **DeepSpeed-Inference** | 43k★, but that's the whole (training) repo; serving layer (MII/FastGen) dormant since mid-2025 | Megatron-style NCCL via `deepspeed.comm` | Low — no pluggability design intent | Low effort, but a dead end | Low — serving layer effectively unmaintained |
| **HuggingFace TGI** | 11k★, **archived read-only 2026-03-21** | N/A — always delegated to vLLM/TensorRT-LLM | Moot | Zero, and zero reason | None — officially dead |

**Conclusion for Part 3**: only vLLM and SGLang are real candidates. Both
have a `GroupCoordinator`-shaped abstraction with an already-proven
pluggable-backend precedent (`NixlConnector`/`mooncake`) — meaning a
well-designed adapter targeting *that* pattern, rather than private-state
mutation, is realistic for both, and the same adapter shape mostly
transfers between them. TensorRT-LLM, llama.cpp, DeepSpeed-Inference, and
TGI are excluded: TensorRT-LLM for architectural incompatibility (comm
baked into a compiled engine), llama.cpp for a fundamentally different,
non-overlapped execution model, and DeepSpeed-Inference/TGI for being
functionally dead as serving platforms.

---

## Part 4 — SGLang deep evaluation

**[verified, live source inspection]**:
- **Scheduler/runtime**: `Scheduler` → `TpModelWorker` → `ModelRunner`, a
  direct structural analog of vLLM's Scheduler/Worker/ModelRunner split.
  Each rank is its own OS process, spawned by SGLang's own launcher
  (`srt/entrypoints/engine.py`), each independently doing TCP-rendezvous
  `torch.distributed` init — same bootstrap shape vLLM uses, meaning no
  fight against an external launcher like `torchrun`.
- **Communication abstraction**: SGLang has its own `GroupCoordinator`
  (`python/sglang/srt/distributed/parallel_state.py`) with the **identical
  branch structure** as vLLM's — `pynccl_comm` if present, else
  `torch.distributed.send/recv`. Strong evidence of direct lineage: SGLang
  ships an actual `monkey_patch_vllm_parallel_state()` function that swaps
  its coordinators into `vllm.distributed.parallel_state` when both
  packages are importable.
- **Not hardcoded to NCCL**: two live precedents for non-NCCL transport
  already exist in SGLang's own tree — a `mooncake`/`mooncake-cpu` backend
  string (routes to an RDMA-based Mooncake transfer engine, used today for
  disaggregated KV-cache transport) and a `naive_distributed.py` filesystem
  -rendezvous coordinator with no NCCL/torch.distributed involvement at
  all.
- **PP status**: real, but explicitly described by the SGLang team as
  immature ("not perfect," pipeline bubbles, "last-rank straggler"
  problem — open roadmap issue). A January 2026 "Chunked Pipeline
  Parallelism" rework is landing now, meaning the exact code this project
  would hook is **actively moving**, not stable legacy code.
- **Insertion points, named concretely**:
  1. `python/sglang/srt/distributed/parallel_state.py`,
     `GroupCoordinator.send()`/`.recv()` — add a transport branch
     alongside the existing `pynccl_comm`/`torch.distributed` fork,
     mirroring the `mooncake` precedent.
  2. `python/sglang/srt/managers/scheduler_pp_mixin.py` (1,652 LOC) — the
     actual PP orchestration (`_pp_send_pyobj_to_next_stage`,
     `_pp_recv_typed_dict`, `_pp_commit_comm_work`), built around a
     `P2PWork` dataclass expecting an async-completion (`.wait()`)
     contract. A custom transport's async handle needs to satisfy this —
     a real design constraint vLLM's simpler blocking `GroupCoordinator`
     doesn't impose.
- **Effort estimate**: **3–5 engineer-weeks** for an SGLang adapter of
  comparable scope to the current vLLM one. Roughly 1 week to port the
  `GroupCoordinator`-level hook (structurally near-identical to vLLM's),
  2–3 weeks for the `P2PWork`-compatible wrapper and bug-fixing pass
  against SGLang's actively-changing PP orchestration layer, plus ongoing
  rebase cost since `managers/` is visibly still growing new files
  (`hisparse_coordinator.py`, `prefill_delayer.py` appeared recently —
  this directory is not settled).

**Would SGLang actually be a *better* base than vLLM?** No — not on its
own, today. It is not more stable (PP is explicitly less mature and more
actively rewritten than vLLM's), not more popular (31k★ vs 88k★, smaller
core team), and would cost 3–5 weeks of fresh integration work to reach
where the vLLM fork already is. Its value is specifically as a **second
adapter target that validates the abstraction**, not as a replacement
for vLLM — its structural similarity to vLLM's `GroupCoordinator` is
exactly why building a runtime whose adapter interface is informed by
*both* frameworks' actual shapes (not just vLLM's) produces a genuinely
reusable interface rather than an vLLM-shaped one with SGLang bolted on
badly.

---

## Part 5 — Should the transport become a standalone reusable runtime?

Yes, and the cost of doing so is lower than it would normally be, for a
concrete, evidenced reason: **64% of the existing transport code already
has zero framework imports** (`base.py`, `factory.py`,
`tcp_transport.py`, `udp_transport.py`, `tensor.py` — 657 of 1,022 LOC).
This is not a hypothetical extraction — it's close to already being
architecturally separate; what's missing is (a) packaging it as an
independent unit and (b) replacing the vLLM-specific
`pipeline_bootstrap.py`/`pp_worker.py` (365 LOC) with a formal adapter
interface instead of direct `parallel_state.py` mutation.

Converged shape, after two rounds of review (this supersedes the first
draft's flatter `TransportRuntime` sketch — the original conflated pipeline
topology, tensor serialization, and control messages into one surface):

```
Backend                 UDPTransport / TCPTransport / (future) QUICTransport
  ↑ already exists: base.py's abstract Transport + factory.py's
    get_transport(backend) dispatch — this layer needed no new design.

Codec                    interface; TensorCodec is the default implementation
  ↑ already mostly exists (tensor.py's serialize_tensor/deserialize_tensor);
    needs to be exposed as a formal Codec interface so ProtobufCodec/
    JSONCodec/etc. can be added later without touching Backend or Connection.

Connection / ConnectionManager   (rename of "Session")
  ↑ one Connection = one point-to-point link, backend-agnostic, owns
    liveness/reconnect. Runtime holds a flat registry (peer_id -> Connection)
    — NOT a topology graph. Two Connections per logical peer-edge: one
    control-plane (reliable/ordered — TCPTransport fits today, no new
    backend needed), one data-plane (throughput — UDPTransport).

FrameworkAdapter (interface), split into two concerns per framework —
  boundary corrected by Phase 2B's synthetic dispatch exercise (see
  transport_runtime/examples/synthetic_dispatch/test_dispatch.py): the
  first draft grouped "scheduling-step dispatch" with lifecycle by
  concept-name; the exercise showed dispatch is operationally identical
  to Communication (both are `Connection.send()`/`.recv()` called
  repeatedly on the hot path), while only bootstrap is a genuinely
  distinct, one-time, cold-path concern:
  ├─ Communication Adapter    # hooks the framework's send/recv seam, AND
  │                           # scheduling-step dispatch — both are just
  │                           # repeated Connection.send()/recv() calls,
  │                           # proven interchangeable by the synthetic
  │                           # dispatch exercise, not assumed
  │    VLLMAdapter:  object.__new__(GroupCoordinator) singleton mutation
  │                  — confirmed structurally necessary, not a shortcut,
  │                  by Phase 3's investigation of the real constructor
  │                  (see Part 7 Phase 3) — now consuming transport_runtime
  │                  instead of vendored vllm/transport/*.py duplicates
  │    SGLangAdapter: GroupCoordinator.send/recv branch + P2PWork wrapper
  └─ Lifecycle Adapter   # hooks process/worker bootstrap ONLY — one-time,
       cold-path (vLLM's TransportPPWorker.init_device() is the existing
       example). The unbuilt MultiprocExecutor cross-machine dispatch gap
       (§1.4) is now understood to be a Communication Adapter concern,
       not a Lifecycle one — it needs a control-plane Connection carrying
       scheduler_output messages, not a new abstraction.
```

Note what's deliberately absent: no `Topology`/`register_pipeline()` object,
no node capability/resource graph. Both were in the first draft and were
cut after review — see Non-Goals above. A `ConnectionManager` that only
tracks live point-to-point links, with no opinion on the shape connecting
them, avoids the dual-source-of-truth failure mode this project already hit
once for a different reason (Bug 2/3 in `README_LIVE_DEPLOYMENT_LOG.md`:
two representations of the same tensor name needing manual reconciliation
— the same class of bug reappears if the runtime keeps its own topology
model alongside whatever rank/topology concept the framework already has).

**Is this significantly more future-proof?** Conditionally yes:

- It is more future-proof **against vLLM's own churn**, because a vLLM
  internal change only requires updating `VLLMAdapter`, not the hole-punch
  protocol, the signaling server, or the wire format — those become
  provably framework-independent, tested independently, and stop being
  collateral damage of vLLM's release cadence.
- It is more future-proof **against framework lock-in specifically**,
  which is the concern the user named directly.
- It is **not** a free future-proofing move against the deeper problem in
  §1.4/§4: framework-specific *orchestration* (multi-machine scheduler
  dispatch in vLLM, `P2PWork`-shaped async completion in SGLang) sits
  above the transport layer and must be built and maintained per
  framework regardless of how clean the transport abstraction is. The
  runtime removes duplicate low-level plumbing; it does not remove the
  need for a real adapter per framework.

---

## Part 6 — Recommendation

**Extract the transport layer into a standalone, framework-agnostic
runtime, keep the current vLLM integration as its first adapter (its
`object.__new__` private-singleton-mutation pattern retained —
confirmed structurally necessary by Phase 3's investigation, not a
shortcut — but now consuming the extracted, tested `transport_runtime`
package instead of vendored duplicate code), and treat SGLang as the
second adapter that proves the abstraction — not as a replacement for
vLLM.**

This is a single direction, not a hedge. Justification, weighing the
evidence above:

1. **The extraction is unusually cheap here**, which is the deciding
   factor. This isn't a generic "abstractions are good" argument — 657 of
   1,022 transport LOC (64%) already have zero vLLM imports, and the
   `GroupCoordinator` shape SGLang independently converged on
   (confirmed via live source read, not assumption) is close enough to
   vLLM's that the adapter pattern transfers with an estimated 3–5 weeks
   of work, not a rewrite. If this extraction cost 3–6 months instead, the
   recommendation would be different — continue the vLLM fork and eat the
   maintenance risk, because a research project with one operator can't
   afford a multi-month refactor for a hypothetical second framework.
   That's not the case here.
2. **Corrected by Phase 3's actual implementation** (see Part 7): the
   first draft of this point claimed rebuilding against `CommunicatorBase`
   would cost "about the same" as the `object.__new__` pattern and should
   be done. That was wrong — `CommunicatorBase` communicators live inside
   an already-rendezvoused process group, and `GroupCoordinator.__init__`
   has no way to reach that state across a NAT boundary without itself
   being the rendezvous mechanism, which is exactly what `object.__new__`
   singleton mutation exists to route around. Building a real replacement
   would mean a custom `transport_runtime`-backed `torch.distributed.
   Store`, a substantially bigger project, not attempted here. What Phase
   3 actually did — and what remains real, evidenced value — is
   decoupling the `object.__new__` pattern's transport dependency from
   the vendored, duplicate `vllm/transport/{base,tcp_transport,
   udp_transport,tensor}.py` onto the tested, bug-fixed
   `transport_runtime` package, verified regression-free on real GPU
   hardware. Bug 5's class of failure (undocumented attribute contracts
   on a synthetic object) is **not** eliminated by this — it was never
   going to be, once the real constructor path turned out to be
   unreachable — and remains a standing risk of the `object.__new__`
   pattern itself, not something this recommendation over-claims to have
   fixed.
3. **The maintenance evidence in Part 1 is concrete, not speculative**:
   an open RFC actively generalizing the exact seam this project hooks, a
   real historical precedent (V0→V1) of a full rewrite of this area, and
   an untracked fork history with no way to diff against upstream. This
   is not "vLLM might change someday" — it's "the specific module this
   integration depends on is under active redesign right now."
4. **This does not throw away the working code.** All five real,
   GPU-verified bugfixes (`moe_wna16.py`, `gpt_oss.py`, `int_wna16.py`,
   `pipeline_bootstrap.py`) stay exactly as they are — they're
   model-loading/quantization correctness fixes, orthogonal to this
   decision, and ship regardless of which direction is chosen. Only the
   PP-transport wiring pattern changes.
5. **What this recommendation explicitly does NOT do**: it does not
   claim SGLang is a better inference engine than vLLM (it isn't, by the
   evidence in Part 4 — smaller, less mature PP, less production
   maturity). It does not claim the Humming blocker gets solved by this
   move (it doesn't — that's an orthogonal quantization-kernel problem,
   see §1.3). And it does not eliminate framework-specific work — the
   `MultiprocExecutor` cross-machine dispatch gap (§1.4) and SGLang's
   `P2PWork` wrapper (§4) still have to be built, adapter or not.

If the user's actual near-term goal were narrowly "get GPT-OSS-120B
generating tokens across this specific 3-machine cluster as fast as
possible," continuing the vLLM fork directly would be the right call —
the Humming blocker is closer to solved on that path (4 concrete forward
options already scoped) than starting a second framework's PP
orchestration from scratch. But the user's stated goal is explicitly
*not* that — it's avoiding long-term lock-in to a heavily patched fork —
and the evidence above supports that instinct on independent technical
grounds, not just because it's what was asked.

---

## Part 7 — Migration roadmap

Each phase is independently shippable; Phase 5 (real deployment) does not
block on Phase 4 (SGLang adapter) — they can run in parallel once Phase 2
is done, since Phase 5's blocker (Humming/quantization) is orthogonal to
which framework adapter is in use. Phase 2 is split into 2A/2B specifically
so there is always something runnable at each milestone, rather than a
single multi-week phase with no intermediate checkpoint.

### Phase 1 — Current fork stabilization
- Pin the exact upstream vLLM commit this checkout tracks (diff against a
  freshly cloned vLLM at the matching `torch==2.13.0` era to reconstruct
  provenance retroactively — this is achievable now, gets strictly harder
  every month the fork drifts further).
- Extract the 5 live bugfixes as a clean, reviewable patch set, separate
  from the vendored copy.
- **LOC**: ~0 new code (documentation/provenance work). **Complexity**:
  low. **Risk**: low. **Value**: high — this is the one item that gets
  more expensive the longer it's deferred.

### Phase 2A — Extract mechanism: backend, codec, connection manager — **DONE**
Implemented at `/kaggle/working/transport_runtime/` (sibling to this
checkout), `vllm/transport/` left completely untouched. What shipped,
diverging from the original estimate only in the details review flagged:

- `backend.py`: `Backend` ABC with an **explicit** error/liveness
  contract (`ConnectionClosedError` vs. `TimeoutError`, with an honest
  documented asymmetry — TCP can detect a real peer-side close, UDP only
  guarantees the signal for a *local* `close()`) — this direct addresses
  Blocking Issue #1 from architecture review (no API had been written
  down before implementation started).
- `ConnectParams` + `TCPBackendConfig`/`UDPBackendConfig`: the original
  `TransportConfig` (one flat dataclass mixing TCP and UDP fields) is
  split so `TCPBackend` cannot see UDP fields and vice versa — addresses
  Blocking Issue #2 (the config-shape leak found by re-reading the
  already-existing `pipeline_bootstrap.py:217-227` call site).
- `backends/tcp.py`, `backends/udp.py`: ported 1:1 from the proven
  `tcp_transport.py`/`udp_transport.py` logic.
- `codec.py`: `Codec` protocol, `TensorCodec`, `JSONCodec`, `BytesCodec`.
- `connection.py`: `Connection` + `ConnectionManager` — flat
  `peer_id -> Connection` registry, no topology graph, per Non-Goals.
- `factory.py`: `get_backend()`/`register_backend()` — new backends
  (QUIC, ...) register without editing this file.

**Exit criterion — swap test**: `tests/test_swap.py` runs the identical
call-site code (`_run_echo`) against `backend_name="tcp"` and
`backend_name="udp"` (the latter against a real signaling server,
skipped automatically if none is reachable). **Passing, 15/15 including
both variants.**

**Two real bugs found and fixed during extraction** (inherited from the
original vLLM-fork code, never exercised by its tests, caught by writing
this package's own test suite before considering the extraction done):
1. `TensorCodec.decode()` crashed on empty tensors (`torch.frombuffer`
   rejects a zero-length buffer) — fixed with a direct `torch.empty()`
   path for that case.
2. A real race in `TCPBackend.close()`: if this backend's own recv
   thread was blocked in `sock.recv()` on the same fd, `close()` alone
   did not reliably send FIN to the peer (a known Linux close/recv race
   across threads) — the peer's `recv()` could hang forever. Fixed with
   `shutdown(SHUT_RDWR)` before `close()`. Caught by
   `tests/test_liveness.py`, itself added because architecture review's
   non-blocking improvement #6 asked for a failure-injection test.

**LOC actual**: ~950 (backend/factory/codec/connection ~450 new,
tcp/udp backends ~430 ported near-verbatim). **Risk realized**: low, as
predicted — both bugs found were pre-existing latent bugs surfaced by
finally writing tests for this exact code path, not new bugs introduced
by the extraction itself.

### Phase 2B — Adapter split and framework-agnosticism proof — **DONE**
Implemented at `transport_runtime/examples/`:

- `plain_pytorch_pipeline/`: a real 3-stage pipeline (Stage A → B → C,
  3 real subprocesses on loopback TCP) computing `x @ W_A @ W_B @ W_C`
  through nothing but `ConnectionManager`/`Connection`/`TensorCodec`.
  **Litmus test passes as an enforced, automated check** — `test_demo.py`
  greps `stage_runner.py`'s source for `GroupCoordinator`/`Worker`/
  `PP rank`/`pp_rank`/`send_tensor_dict`/`vllm`/`sglang` and fails the
  suite if any appear, not just a one-time manual read. **Correctness is
  verified numerically**: the pipeline's real output is compared against
  a directly-computed (non-pipelined) ground truth with the same seeds,
  not just "the processes didn't crash."
- `synthetic_dispatch/`: a fake framework's bootstrap-once/dispatch-5×
  loop, built to answer the open design question from review's
  non-blocking improvement #1 (is "Communication + Executor/Lifecycle"
  the right adapter split?). **Result: no.** The exercise showed
  scheduling-step dispatch needs nothing beyond ordinary
  `Connection.send()`/`.recv()` called repeatedly — operationally
  identical to Communication, not to one-time bootstrap. Part 5's
  diagram above is corrected accordingly: **Communication Adapter**
  (send/recv AND dispatch, both hot-path) vs. **Lifecycle Adapter**
  (bootstrap only, cold-path) — not "Communication vs.
  Executor/Lifecycle" as first drafted.
- **≥90% LOC retention**: currently **vacuously 100%** — no framework
  adapter exists inside `transport_runtime` at all yet (by design; see
  Non-Goals — adapters live in each framework's own checkout). This
  metric becomes meaningful once Phase 3 exists to measure against;
  stated here rather than claimed as a false victory.

**LOC actual**: ~330 (stage_runner.py + run_demo.py + test_demo.py +
test_dispatch.py combined). **Risk realized**: low — the one real finding
(the adapter-split boundary was wrong as first drawn) is exactly the kind
of thing this checkpoint existed to catch before Phase 3, not a surprise
that threatens the overall design.

### Phase 3 — vLLM Communication + Lifecycle adapter — **DONE, scope corrected by real investigation**

**Important correction, found only by reading the real constructor, not
assumed**: Part 1/6 above recommended migrating to a `CommunicatorBase`
-conformant adapter via `GroupCoordinator`'s real `__init__`, as the
"sanctioned" alternative to `object.__new__` singleton mutation. Having
now actually read `GroupCoordinator.__init__`
(`vllm/distributed/parallel_state.py:484-560`), that recommendation was
based on an incomplete premise: the real `__init__` **unconditionally**
calls `torch.distributed.new_group(ranks, backend=..., timeout=...)` for
both the device group and the CPU group — a real collective operation
requiring every rank to already share a rendezvoused default process
group. `torch.distributed.new_group()` does not accept a custom `Store`
(checked its signature directly); only `init_process_group()` does. So
the actually-sanctioned path would require building a full
`torch.distributed`-compatible `Store` backed by `transport_runtime`
(implementing `set/get/add/wait/compare_set/...` correctly over a
best-effort transport) — a substantially larger project than the
~400-600 LOC this phase originally budgeted, and out of scope for this
pass. `tests/transport/_pipeline_shim.py`'s own docstring independently
confirms this same constraint ("`GroupCoordinator.__init__`
unconditionally calls [`new_group`]... this phase's constraints forbid
[it]") — this project's own earlier work had already hit and documented
the same wall.

**What this means**: `object.__new__` singleton mutation is not a lazy
shortcut avoiding a known-better pattern — it is *structurally required*
for a cross-NAT PP group, given vLLM's real constructor's hard
rendezvous requirement. Phase 3's scope was corrected accordingly: keep
the `object.__new__` pattern (now justified in code comments, not left
unexplained), and migrate what's actually achievable — decoupling
`pipeline_bootstrap.py`/`parallel_state.py`'s transport usage from the
vendored `vllm/transport/{base,tcp_transport,udp_transport,tensor}.py`
onto the tested, bug-fixed `transport_runtime` package instead.

**What shipped**:
- `vllm/transport/pipeline_bootstrap.py`: `establish_pp_transports()`
  rebuilt on `transport_runtime.get_backend()`/`ConnectParams`/
  `TCPBackendConfig`/`UDPBackendConfig` instead of the old
  `vllm.transport.get_transport()`/`TransportConfig`. `TYPE_CHECKING`
  import updated to `transport_runtime.backend.Backend`.
  `install_transport_pp_group()` itself needed **zero changes** — it was
  already fully duck-typed (confirmed via `grep` — no `isinstance`
  checks anywhere on the transport object), so any object exposing
  `send(bytes)`/`recv(timeout)`/`close()` works, which `transport_runtime.
  Backend` does by construction (ported faithfully from the same
  interface).
- `vllm/distributed/parallel_state.py`: the `TYPE_CHECKING` import,
  `_transport_send_tensor_dict`/`_transport_recv_tensor_dict`, and the
  single-tensor `send()`/`recv()` methods' transport branches now use
  `transport_runtime.codec.TensorCodec` instead of
  `vllm.transport.tensor.serialize_tensor`/`deserialize_tensor`/
  `TransportProcessGroup` — wire-format-identical (ported faithfully),
  so this is a pure backend swap, not a protocol change. The
  single-tensor `send()`'s pre-existing lack of CPU-staging (unlike the
  dict path, which already staged) was preserved exactly as-is —
  deliberately not fixed here, since Phase 3's scope is migration, not
  new behavior.
- `vllm/transport/{base,factory,tcp_transport,udp_transport,tensor}.py`
  left **untouched** — `tests/transport/test1`-`test9` (transport-layer-
  only tests) still import them directly and still pass; removing them
  is a separate cleanup not attempted this pass.

**Verified on real hardware** (this session's environment has 2× real
Tesla T4 GPUs, confirmed via `nvidia-smi`) — every test that exercises
the code paths this phase touched was re-run after the migration, not
just byte-compiled:
- `test18_real_bootstrap_pp.py` — both `--transport tcp` and `--transport
  udp`: **PASS**, identical checkpoints to the pre-migration baseline
  (real local torch.distributed TP group + real `GroupCoordinator.
  send_tensor_dict`/`recv_tensor_dict` over the new backend).
- `test20_real_bootstrap_pp_three_stage.py --transport udp` (3-stage,
  middle-rank dual `transport_prev`/`transport_next` links, real UDP
  hole punch): **PASS**.
- `test11`-`test16` (the `_pipeline_shim.py`-based tests exercising the
  exact `GroupCoordinator.send()`/`.recv()` single-tensor methods this
  phase modified): **PASS**, all 6.
- `test1`-`test9`, `test17` — the former are untouched code paths
  (sanity-checked, unaffected); `test17` (real GPT-2 model load) was
  **not completed** — it needs a local `/models/gpt2` checkpoint this
  environment doesn't have, an environment/data gap unrelated to this
  migration, not a regression. Stated honestly rather than silently
  skipped.

**LOC actual**: ~90 lines changed across the 2 files (far less than the
~400-600 originally budgeted for a `CommunicatorBase` rebuild — because
that rebuild turned out to be infeasible this pass, and the migration
actually done is a narrower, mechanical backend swap). **Risk realized**:
low — every test that could regress was re-run on real GPU hardware and
passed; the only real finding was the corrected understanding of why
`object.__new__` exists at all, not a new problem.

### Phase 4 — SGLang adapter — **partially done, real-hardware-verified, scope corrected**

Cloned real SGLang (`sgl-project/sglang`, `main` branch) into
`/kaggle/working/sglang/` and verified the earlier web research directly
against source (some of it held up, some didn't):

- **Confirmed accurate**: `GroupCoordinator` at
  `python/sglang/srt/distributed/parallel_state.py`, near-identical
  branch structure to vLLM's, `send`/`recv`/`send_tensor_dict`/
  `recv_tensor_dict`/`send_object`/`recv_object` all present roughly
  where expected. `mooncake` backend precedent and `naive_distributed.py`
  both real.
- **Corrected**: stock SGLang has **no** `transport` attribute or hook
  anywhere in `GroupCoordinator` — unlike vLLM's checkout (which already
  had one from earlier project work), this had to be built from scratch,
  not migrated.
- **Confirmed independently** (not assumed from vLLM's case): SGLang's
  `GroupCoordinator.__init__` also unconditionally calls
  `torch.distributed.new_group()` — the identical hard constraint found
  for vLLM in Phase 3. `object.__new__` bypass is therefore structurally
  necessary here too, not a vLLM-specific quirk. This was verified by
  reading the source directly, not inferred from lineage.

**What shipped, in the cloned checkout**:
- `python/sglang/srt/distributed/parallel_state.py`: added `transport`/
  `transport_prev`/`transport_next` class attrs + `__init__` param;
  early-exit transport branches in `send()`, `recv()`, `send_tensor_dict()`,
  `recv_tensor_dict()`, using `transport_runtime.codec.TensorCodec` (the
  same helpers already proven for vLLM, ported not reinvented).
- `python/sglang/srt/distributed/transport_bootstrap.py` (new): direct
  port of vLLM's `pipeline_bootstrap.py` onto SGLang's `GroupCoordinator`.

**Verified on real hardware, not just byte-compiled** (this session's
2× Tesla T4, same environment as Phase 3): a real `test18`-equivalent —
2 processes, each forming a real local SGLang `torch.distributed` group
via SGLang's own unmodified `init_distributed_environment`/
`ensure_model_parallel_initialized`, then swapping in the transport-backed
PP group and exchanging a real tensor via `send_tensor_dict`/
`recv_tensor_dict` — **PASSES for both TCP and UDP** (real hole punch).
See `sglang/tests/transport_runtime/` for the test and its README.

**Real environment finding, not a sandbox quirk**: getting SGLang
importable at all required two shims (`sglang/tests/transport_runtime/
_env_shim.py`, same pattern as vLLM's `_env_stubs.py`) — a `transformers`
version conflict (`qwen3_asr` double-registration), and, more
significantly, **this environment's PyPI `sgl_kernel` wheel only ships a
compiled SM100 (Blackwell) binary, no SM75 (Tesla T4) variant** — the
import crashes on real T4 hardware before any kernel is ever called.
`sgl_kernel` had to be stubbed to get past it. This is a real, evidenced,
T4-specific packaging gap in SGLang's own kernel distribution — see Phase
5 below for why this matters beyond just this test.

**Honest scope — not attempted**: `send_object`/`recv_object` are not
hooked (bypassed entirely by the `send_tensor_dict`/`recv_tensor_dict`
early-exit, matching vLLM's pattern, but anything calling them directly
still uses real `torch.distributed`); `async_send=True`/the `P2PWork`
async-completion contract is asserted against, not supported (the one
real API-shape difference from vLLM flagged in the original Part 4
research, still unaddressed); no real model loading was exercised (this
is `test18`-equivalent scope only, not `test17`/`test19`/`test20`-
equivalent). Full parity with vLLM's Phase 3 would need those three,
genuinely more work, not attempted this pass.

**LOC actual**: ~230 lines changed/added across `parallel_state.py` (+~90)
and `transport_bootstrap.py` (new, ~175), plus ~350 lines of test +
shim + README. Well under the original 200–700 LOC estimate for the
transport-glue portion; the `P2PWork` wrapper (est. 100–300 LOC) was not
built, consistent with the honest scope above. **Risk realized**:
lower than estimated for the glue itself (SGLang's real lineage
similarity to vLLM held up under direct verification, not just web
research); the `sgl_kernel`/T4 packaging gap was an unplanned, real
finding that took real investigation time but didn't block the actual
transport-hook goal once stubbed.

### Phase 5 — Real GPT-OSS deployment — **both known SM75 kernel bugs fixed and verified; model-level load still untested**

This session's environment turned out to have real, persistent state from
the original deployment: the actual 61GB GPT-OSS-120B GPTQ checkpoint
(`/data/models/gpt-oss-120b-gptq`, verified complete, 16/16 shards), the
real `humming-kernels==0.1.10` package installed, and 2× real Tesla T4
GPUs. This made the "patch humming's tuning table" path (option 4 of the
four originally scoped) actually attemptable for real, not just
theoretical — and revealed it was more tractable, and more subtle, than
the original diagnosis suggested.

**Correction to the original diagnosis**: `humming` was previously
described (`README_LIVE_DEPLOYMENT_LOG.md`) as "closed-source." On direct
inspection this session, it is not — the Python orchestration layer and
the CUDA kernel templates (`.cuh` files, compiled via NVRTC at *runtime*
from plain-text source) are both fully readable. This materially changes
what's possible: the tuning heuristic that selects a kernel's tile shape
is ordinary, patchable Python.

**Bug 1 — root-caused precisely, fixed, and verified** (full writeup:
`/kaggle/working/humming_fix/README.md`). `humming/tune/base.py`'s MoE
`block_shape_m` heuristic can select a non-32-aligned value (16 or 48)
without checking whether the base config it's overriding already
requested `num_write_splits=2` — exactly the combination
`humming/include/humming/epilogue/pipeline.cuh`'s
`static_assert(BlockShape::M % 32 == 0)` (gated by
`if constexpr (kNumWriteSplits > 1)`) rejects. This is the literal
static_assert from the original crash. Fixed via a runtime monkeypatch
(`humming_fix/patch.py`) — deliberately not an in-place edit to the
installed package, since that's outside this project's own working
directory (a real permission boundary this session respected rather than
routing around). **Verified two ways**: (1) a full sweep of GPT-OSS's
real shapes across `shape_m ∈ {1,8,48,96,128,256}` — every value hit the
bad combination before the fix, none after (`humming_fix/test_repro.py`,
6/6 passing); (2) confirmed this exact static_assert no longer fires in a
real NVRTC compile attempt.

**Bug 2 — found, precisely located, and (in the following pass) FIXED and
verified.** Attempting a *full* real NVRTC compile (not just checking the
heuristic) with Bug 1's fix applied surfaced a second, independent SM75
incompatibility: `humming/kernel/humming.py`'s `select_mma_op_class()`
selects an `m16n8k16` MMA instruction for 16-bit inputs (GPT-OSS's
activation dtype), which real T4 hardware rejects (`ptxas: Feature
'.m16n8k16' requires .target sm_80 or higher`) — confirmed to reproduce
identically whether or not Bug 1's fix is applied, i.e. a genuinely
separate bug, not a symptom of the first. **A first attempted fix was
tried and confirmed wrong** (mirroring the neighboring `sm75+int8:
mma_shape_m=8` special case, producing `m8n8k8`, itself an invalid PTX
modifier) — removed rather than shipped.

**The real fix**, found by empirically probing SM75's actual Tensor Core
capability instead of pattern-matching a neighboring case: a minimal
raw-PTX probe compiled directly with `nvcc -arch=sm_75`
(`sm75_mma_probe/probe.py`, independent of `humming`) showed `m16n8k16`
is rejected but `m16n8k8` (same M=16, only K halved) compiles cleanly.
Keeping `mma_shape_m=16` and only setting `mma_shape_k=8` needed **zero
C++ changes** — `humming/config/mma.py`'s PTX/register-count generation
and `humming/include/humming/mma/wmma.cuh`'s `WMMA::run()` K-loop are
both already fully generic over MMA shape, confirmed by reading the
source directly. Verified three ways: (1) the generated PTX for
`MmaOpClassImpl(16,8,8,...)` is byte-identical to what the probe
confirmed compiles; (2) a real `HummingKernel.prepare_kernels(...)` call
with GPT-OSS's exact MoE shape now compiles **and** `cuModuleLoad`s
successfully (previously the exact failure point); (3) a real dense
fp16×int4 GEMM run end-to-end through `humming.ops.humming_gemm()` on
real T4 hardware produces output within int4-quantization-noise
tolerance of a float32 reference (mean abs err 0.0137) — i.e. numerically
correct, not just compiling. Full writeup: `humming_fix/README.md`;
tests: `humming_fix/test_repro.py`,
`test_bug2_fixed_gpt_oss_moe_shape_compiles_and_loads`;
`humming_fix/test_correctness.py`.

**Net result**: both known SM75/T4 kernel-level blockers for GPT-OSS-120B
are now fixed and verified at the kernel-compile + numerical-correctness
level.

**Follow-up: real single-layer engine-level load, done.** The full
61GB checkpoint cannot fit in this sandbox's 2×T4 (32GB total VRAM)
regardless of kernel bugs — that's a hard memory ceiling, not something
either fix addresses (it's exactly why this project's actual target is
multi-machine pipeline parallelism). To get real signal anyway: extracted
layer 0's actual tensors (real GPTQ `qweight`/`qzeros`/`g_idx`/`scales`,
real `group_size=64`, real shapes) plus the model's global tensors
(embeddings, final norm, lm_head) from the real checkpoint's shard 1 into
a standalone ~4GB single-layer checkpoint, and ran vLLM's **actual**
production model-loading path end-to-end via `LLM(..., moe_backend=
"humming")` — not a hand-rolled repro. Real result: `Using 'HUMMING'
WNA16 MoE backend` selected automatically, real weights loaded, real
forward pass, real token generation, zero crashes, zero NaN-path
failures — the exact call chain (`auto_gptq.py` → `humming_utils.py` →
`humming_fix`'s two patches → real NVRTC compile → real kernel launch)
that used to hit both bugs now completes cleanly on real hardware.
Output text is gibberish, as expected and not a correctness concern —
34 of 36 real transformer layers are structurally absent by construction
in this single-layer probe, so incoherent output is exactly what a
correctly-functioning stub should produce; the signal here is "no crash,
no NaN, real kernels ran," not "coherent generation."

**A real mistake happened during this and was corrected**: reusing a
symlink path for `model.safetensors.index.json` inside the scratch
single-layer directory caused a `open(path, "w")` to follow the symlink
and overwrite the **real checkpoint's** index file in
`/data/models/gpt-oss-120b-gptq/`, truncating it from 46983 keys to
1308. Caught immediately by inspection; fully recovered by reconstructing
the index from the 16 real shard files' own self-describing headers (no
tensor data was ever touched, only the index was affected) — verified
restored to all 46983 keys, all 36 layers × 128 experts present,
`lm_head`/embeddings/norm intact. Noted here as a reminder that writing
through any symlinked path in a shared data directory is exactly this
kind of risk, even when the intent is scratch-local.

**Not yet attempted**: loading more than one real layer (blocked by this
sandbox's VRAM, not by anything code-related), every `GemmType` actually
used (only `grouped_contiguous` exercised), `bf16` activations, and
performance measurement. The realistic path to a genuine full-scale
validation remains multi-machine pipeline parallelism (this project's
actual target architecture), not a bigger single node.

**Separately**: the earlier "switch to SGLang to sidestep Humming"
question is answered, not open (Phase 4 found SGLang depends on
`humming-kernels[cu13]==0.1.10` directly, plus has its own unrelated
`sgl_kernel`/T4 gap) — switching frameworks would not avoid this class of
problem.

- Remaining, not attempted this session: close the `MultiprocExecutor`
  cross-machine scheduler-dispatch gap (~400–700 LOC, already scoped in
  `README_RUN_GPTOSS_CLUSTER.md`) — Phase 2B's synthetic dispatch
  exercise clarified this is **Communication Adapter** work (a
  control-plane `Connection` carrying `scheduler_output` messages), not
  a new abstraction — the primitive it needs already exists and is
  tested in `transport_runtime`. No longer blocked on Humming/T4 at the
  kernel level, but still not attempted — full model load hasn't been
  proven yet either (see above), so this remains sequenced after that.
- **LOC actual**: ~130 total across both fixes (Bug 1's generalized
  post-check + Bug 2's `select_mma_op_class` override), plus their
  verification tests and a raw-PTX probe script. **Risk note**: this was
  flagged as "the one phase whose outcome isn't under this project's
  control" — that held for the diagnostic work (real domain-specific
  investigation was required, not just engineering), but resolved via
  empirical probing (compiling raw PTX variants directly against the
  real GPU) rather than needing external reference material, once the
  actual failing instruction shape was identified precisely.

**Addendum, 2026-08-03 — session reset and reconstruction**: the working
directory that produced everything above (this file included) was lost
to a session reset before it could be backed up (root cause: a large
download had landed under `/kaggle/working`, whose small dedicated
volume filling up is what actually kills the session — not a generic
"reset wipes everything" event; corrected project memory accordingly).
`vllm/` and `udp_holepunch/` survived (they were already under
`/kaggle/working`, small). `humming_fix/`, `transport_runtime/`,
`sm75_mma_probe/`, and `/data/models/gpt-oss-120b-gptq` did not.

Recovery, in order of what was actually possible:
1. `sm75_mma_probe/probe.py` had no surviving copy anywhere — rebuilt
   from a pasted terminal log of the original session and re-run for
   real; produced byte-identical `ptxas` verdicts (`m16n8k16` rejected,
   `m16n8k8` compiles) to what the log recorded.
2. Two sibling machines from the real 3-machine cluster this project
   targets ("Akun 2"/"Akun 3") turned out to still be reachable over SSH
   and still held the real, unlost `humming_fix/`, `transport_runtime/`,
   and the full 61GB checkpoint — pulled the genuine original files from
   there rather than trusting a from-log reconstruction, and confirmed
   they materially improved on the from-log rebuild (the real Fix 1
   handles a second assert, `BlockShape::M == WarpShape::M` on the dense
   branch, that the log-based reconstruction of Fix 1 had missed
   entirely — found via a real dense-shape correctness run in the
   original session, not visible from the log excerpt alone).
3. Re-ran `humming_fix`'s full test suite (8/8) and `transport_runtime`'s
   full suite (11/11 + both example demos) for real on this session's
   own 2× T4 - both packages work unmodified in the freshly-reset
   environment.
4. Re-did the "single-layer engine-level load" follow-up for real (see
   `humming_fix/single_layer_probe/`), on Akun 2 rather than locally
   (its vLLM checkout already has compiled kernels from a real
   `pip install -e .`; this session's own checkout does not, and
   rebuilding that from scratch was correctly judged not worth
   redoing when a working, real build already existed one SSH hop
   away). Reproduced the exact real result: `Using 'HUMMING' WNA16 MoE
   backend`, real weights loaded, real forward pass, real tokens
   generated, zero crashes - plus one new, real, environment-specific
   finding not in the original run (`libnvrtc-builtins.so.13.0` present
   on disk but not on `LD_LIBRARY_PATH`, for the separate
   `RepackWeightKernel` compile path) with its fix documented in
   `single_layer_probe/README.md`.

Net effect: Phase 5 is now re-verified end-to-end against real GPU
hardware and the real checkpoint, using the *actual* original code
(pulled from surviving machines) rather than a best-effort textual
reconstruction, wherever the actual originals could be reached.
`sm75_mma_probe/probe.py` remains a faithful reconstruction (no original
survived to compare against), but its result was independently
re-verified by direct execution, not merely retyped.

### Phase 6 — Performance tuning
- Pipeline bubble reduction; consider SGLang's chunked-PP micro-batching
  idea as a design reference regardless of which framework ends up
  serving; profile real UDP transport throughput/latency across the
  actual WAN links in use.
- **LOC**: moderate, mostly tuning/config, some scheduling code.
  **Complexity**: medium. **Risk**: low architecturally, high in
  engineering time (this is where real-world WAN variance bites).

### Phase 7 — Documentation and publication
- Write up the standalone transport runtime, both adapters, and the
  cluster deployment guides as a coherent whole; consider whether the
  cross-NAT pipeline-parallel result is itself worth a short technical
  writeup/publication given it's a genuinely novel-sounding result
  (**[speculative]** — novelty wasn't independently verified against
  prior art in this session; worth a literature check before claiming
  novelty publicly).
- **LOC**: ~0 code. **Complexity**: low. **Risk**: low.

---

## Appendix: raw research inputs

The full per-framework research report (with live source citations,
GitHub star/contributor counts, and the complete SGLang deep-dive with
file:line-level detail) and the full repository coupling audit (with
every `file:line` integration point) that this document synthesizes were
produced as part of this review and are available in this session's
agent transcripts if a lower-level source is needed; the actionable
conclusions from both are fully incorporated into Parts 1–4 above.
