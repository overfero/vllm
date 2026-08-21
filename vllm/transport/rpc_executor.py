"""Closes the gap documented in `README_RUN_GPTOSS_CLUSTER.md` Task 10
("the Executor RPC / scheduler_output dispatch gap"): today, only the
machine whose `EngineCore` actually runs the real `Scheduler` (the
"driver" - in this project's topology, Machine C, the one serving the
API) ever calls `execute_model()`. The other two machines' local
`MultiprocExecutor` never gets driven for a given step at all, so their
workers' `get_pp_group().irecv_tensor_dict()` blocks forever.

`TransportExecutor` is the driver-side half of the fix. It IS a real
`MultiprocExecutor` for this machine's own local TP workers (nothing
about local dispatch changes - same `rpc_broadcast_mq`/`response_mqs`
shared-memory path, same `collective_rpc` for everything not overridden
below). What it adds: for the four PER-STEP methods `EngineCore.step()`
actually calls every scheduling step (`execute_model`, `sample_tokens`,
`execute_dummy_batch`, `take_draft_token_ids`), it ALSO forwards the same
call, with the same arguments (the real `SchedulerOutput`/`GrammarOutput`
this machine's Scheduler just produced), to every remote stage's
`stage_server.py` process over a dedicated RPC control-channel
`Transport` connection - separate from the per-local-GPU PP
activation-tensor links `establish_pp_transports` already owns, so the
two never contend for the same socket/port.

Deliberately NOT forwarded: `load_model`, `get_kv_cache_specs`,
`determine_available_memory`, `initialize_from_config`,
`compile_or_warm_up_model`, and everything else `collective_rpc` carries
(lora ops, sleep/wake, profiling, ...). Each stage's checkpoint shard,
KV cache spec, and available GPU memory are legitimately different per
machine - `stage_server.py` performs all of that setup independently,
using its own local `EngineCore` construction, exactly like the driver
does for itself. The one piece of global state that MUST agree across
machines - `num_gpu_blocks`, since every `scheduler_output` forwarded
here references block-table ids that only mean the same thing if every
stage sized its KV cache identically - is handled by requiring
`--num-gpu-blocks-override` (a real, existing vLLM flag, see
`vllm/config/cache.py`) to be passed identically on every machine's
launch command, not by anything in this file. See
`pp_tests/BLOCKER_REPORT.md` Blocker 3 for why this matters and why an
override, not runtime coordination, was chosen (lowest-effort real fix
available).

Selected via vLLM's own extensibility point, zero vllm core changes:

    vllm serve $MODEL_PATH \\
        --distributed-executor-backend vllm.transport.rpc_executor.TransportExecutor \\
        --worker-cls vllm.transport.pp_worker.TransportPPWorker ...

(`Executor.get_class`, `vllm/v1/executor/abstract.py`, already resolves
a dotted qualname string for `distributed_executor_backend` via
`resolve_obj_by_qualname` - the same mechanism `--worker-cls` uses.)

---

2026-08-15 microbatching (pipeline-bubble-elimination) experiment, OFF by
default. Two things had to be true together for this to do anything at
all - doing only one is a no-op or worse:

1. vLLM's own `EngineCore.step_with_batch_queue()` (vllm/v1/engine/core.py)
   needs `vllm_config.max_concurrent_batches > 1` to even get selected
   over the plain `step()` this project has used until now - that
   property reads `parallel_config.pipeline_parallel_size`, which this
   project's `stage_server.py`/`launch_pp_stage.py` deliberately keep at 1
   on every machine (the REAL value would make vLLM try to bootstrap a
   real `torch.distributed` PP group across NAT'd hosts inside
   `initialize_model_parallel()` - an immediate crash, not a slow
   timeout). `TransportExecutor._init_executor` below mutates
   `vllm_config.parallel_config.pipeline_parallel_size` to the real
   cross-machine stage count, but ONLY AFTER `super()._init_executor()`
   returns - i.e. only after this machine's own local workers have
   already spawned and read the (unmutated, correct) value of 1 for
   their own local-only `initialize_model_parallel()` call. Workers are
   separate OS processes; a driver-process-only mutation after they've
   already started never reaches them. Scheduler/KV-cache-manager
   construction (`EngineCore.__init__`, both happen strictly after
   `self.model_executor = executor_class(vllm_config)`) then see the
   mutated value, sizing `max_in_flight_tokens`-driven KV block
   reservation correctly for genuinely-overlapping batches instead of
   under-reserving (the latter being exactly the kind of silent
   KV-slot-collision bug class this project has hit before under a
   different name - see `_RemoteStageLink`'s docstring below).

2. Even with (1) done, `step_with_batch_queue()` dispatching step N+1's
   `execute_model`/`sample_tokens` RPCs before step N's replies have
   arrived is USELESS if `_dispatch_remote` still serializes one full
   send+recv round trip per link before allowing the next send (which is
   exactly what the pre-2026-08-15 code below did, deliberately, to close
   a real crossed-response race - see `_RemoteStageLink`'s docstring).
   The fix isn't to relax that serialization back to "hope for the
   best" - it's to give `UDPTransport.recv()`'s shared FIFO queue what it
   was missing: a request/response correlation id, carried in the
   payload itself, so a single background reader thread per link can
   route each arriving reply to the correct pending `Future` regardless
   of arrival order, with NO assumption that only one request is ever on
   the wire. This closes the original race at its actual root cause
   (no way to tell replies apart) instead of working around it by
   forbidding the concurrency that would trigger it - which is what
   real pipelining needs.

Both are controlled by `VLLM_TRANSPORT_ENABLE_PIPELINING=1` (new
correlation-id wire protocol) and `VLLM_TRANSPORT_BATCH_QUEUE_SIZE=N`
(the mutated `pipeline_parallel_size`) - both env vars, both must be set
identically on every stage (driver and every `stage_server.py`), never
partially. See `cluster/qwen35_122ba10b_3machine_pipelined.sh`.
"""
from __future__ import annotations

import collections
import itertools
import os
import pickle
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

import cloudpickle

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.v1.executor.multiproc_executor import MultiprocExecutor

logger = init_logger(__name__)

_ENV_SELF_NAME = "VLLM_TRANSPORT_SELF_NAME"
_ENV_REMOTE_STAGE_NAMES = "VLLM_TRANSPORT_REMOTE_STAGE_NAMES"
_ENV_REMOTE_STAGE_HOSTS = "VLLM_TRANSPORT_REMOTE_STAGE_HOSTS"
_ENV_RPC_PORT = "VLLM_TRANSPORT_RPC_PORT"
_ENV_SIGNALING_URL = "VLLM_TRANSPORT_SIGNALING_URL"
_ENV_CONNECT_TIMEOUT = "VLLM_TRANSPORT_CONNECT_TIMEOUT"
# 2026-08-15 microbatching experiment (see cluster/qwen35_122ba10b_3machine_pipelined.sh).
# Both OFF by default - baseline (cluster/qwen35_122ba10b_3machine.sh) never sets
# these, so it exercises the exact same code path (single-request-in-flight
# lock, plain 1:1 step()) it always has. Only a deliberately-launched
# candidate deployment sets them, and BOTH must be set together on every
# stage (driver AND every stage_server.py) - see module docstring additions
# below for why pipelining without the batch-queue depth increase (or vice
# versa) doesn't accomplish anything.
_ENV_ENABLE_PIPELINING = "VLLM_TRANSPORT_ENABLE_PIPELINING"
_ENV_BATCH_QUEUE_SIZE = "VLLM_TRANSPORT_BATCH_QUEUE_SIZE"
# 2026-08-15 RPC fusion: only meaningful (and only allowed) on top of
# pipelining - see TransportExecutor.execute_model/sample_tokens. Fuses a
# single step's execute_model+sample_tokens into ONE round trip per remote
# link (stage_server.py's execute_model_and_sample_tokens handler) instead
# of two. Independent toggle from pipelining itself so each can be
# benchmarked/rolled back separately.
_ENV_ENABLE_FUSION = "VLLM_TRANSPORT_ENABLE_RPC_FUSION"
_ENV_DEBUG_TIMING = "VLLM_TRANSPORT_DEBUG_TIMING"

_DEFAULT_RPC_PORT = 40000

_STATUS_OK = "ok"
_STATUS_ERROR = "error"


@dataclass
class _RemoteStageLink:
    name: str
    transport: object  # vllm.transport.base.Transport
    timeout: float
    # Root cause of the multi-step decode-degradation bug (garbage output
    # after the first token or so, on every model tested - GPT-OSS-120B AND
    # a plain unquantized Qwen2.5-7B): vLLM's own PP pipelining
    # (`EngineCore.step_with_batch_queue`, vllm/v1/engine/core.py) calls
    # `execute_model`/`sample_tokens` for step N+1 via `non_block=True`
    # *before* step N's result is consumed - that overlap is the whole
    # point of the batch queue. `_dispatch_remote` for both steps then runs
    # on this same `_RemoteStageLink`'s shared `ThreadPoolExecutor`, so two
    # threads could call `transport.send()`/`recv()` on the exact same
    # `Transport` instance concurrently. `UDPTransport.recv()` has no
    # request/response correlation id - it just pops the next fully
    # assembled message off one shared FIFO queue - so a `recv()` call
    # blocked waiting for step N's reply could instead receive step N+1's
    # reply (or vice versa). The remote `stage_server.py` loop processes
    # exactly one request at a time in receipt order and executes it
    # in-place before replying, so any such crossed request/response also
    # means the remote executed `scheduler_output`s out of order relative
    # to the PP activation tensors it exchanges with its neighbor over the
    # *separate* PP transport channel - wrong tokens/positions get written
    # into KV-cache slots that then poison every subsequent step. This lock
    # serializes calls to *this* link only (A and B can still overlap with
    # each other) so at most one request is ever in flight on the wire at
    # a time, matching stage_server.py's strictly sequential processing and
    # eliminating the crossed-response race entirely.
    #
    # 2026-08-15: this lock is still the default (VLLM_TRANSPORT_ENABLE_PIPELINING
    # unset) and completely unchanged in behavior. When pipelining IS enabled,
    # `_dispatch_remote` skips this lock entirely and uses `pending`/`reader_thread`
    # below instead - a request-id-correlated response, which closes the same
    # race a different way (see module docstring) and additionally allows more
    # than one request in flight on this link at once, which the lock forbids
    # by design.
    lock: threading.Lock = field(default_factory=threading.Lock)
    # --- pipelining-only fields below (unused, stay at their defaults, when
    # VLLM_TRANSPORT_ENABLE_PIPELINING is unset) ---
    pipelined: bool = False
    pending: dict[int, "Future[tuple[str, object]]"] = field(default_factory=dict)
    pending_lock: threading.Lock = field(default_factory=threading.Lock)
    id_counter: "itertools.count[int]" = field(default_factory=itertools.count)
    reader_thread: threading.Thread | None = None
    reader_stop: threading.Event = field(default_factory=threading.Event)


def _reader_loop(link: _RemoteStageLink) -> None:
    """Background thread, pipelining mode only: continuously drains this
    link's transport `recv()` queue and routes each `(request_id, status,
    result)` reply to the matching pending `Future` by id - regardless of
    what order replies arrive in, since each one is self-describing now
    (see module docstring). One thread per link, started once at connect
    time and left running for the link's whole lifetime; `_dispatch_remote`
    never calls `transport.recv()` itself in pipelined mode.
    """
    while not link.reader_stop.is_set():
        try:
            raw = link.transport.recv(timeout=1.0)
        except TimeoutError:
            continue
        except (ConnectionError, OSError) as e:
            logger.warning("pipelined RPC reader for %s: transport closed (%r)", link.name, e)
            break
        try:
            request_id, status, result = pickle.loads(raw)
        except Exception:
            logger.exception(
                "pipelined RPC reader for %s: failed to unpack reply, dropping", link.name
            )
            continue
        with link.pending_lock:
            fut = link.pending.pop(request_id, None)
        if fut is None:
            logger.warning(
                "pipelined RPC reader for %s: reply for unknown/already-resolved "
                "request_id=%s (late timeout retry?), dropping", link.name, request_id
            )
            continue
        if not fut.done():
            fut.set_result((status, result))
    # Drain: unblock anything still waiting so callers get a clean error
    # instead of hanging forever if the link died mid-flight.
    with link.pending_lock:
        stale = list(link.pending.items())
        link.pending.clear()
    for _request_id, fut in stale:
        if not fut.done():
            fut.set_exception(ConnectionError(f"RPC link {link.name!r} reader stopped"))


def _connect_remote_stages(pipelined: bool = False) -> list[_RemoteStageLink]:
    """Read `VLLM_TRANSPORT_REMOTE_STAGE_NAMES` (comma-separated, e.g.
    "MachineA,MachineB") and connect one RPC control-channel `Transport`
    to each - the driver always initiates (connects), every
    `stage_server.py` always listens, so there is no ordering ambiguity
    even with 2+ remote stages (Machine B's launch is the only one with
    two PP tensor peers; the RPC control channel only ever exists between
    the one driver and each non-driver stage, so no stage server needs
    more than one incoming RPC connection).

    Returns an empty list (no-op forwarding) if the env var is unset or
    empty - lets `TransportExecutor` also work correctly as a plain local
    `MultiprocExecutor` for single-machine testing.
    """
    self_name = os.environ.get(_ENV_SELF_NAME)
    remote_names_raw = os.environ.get(_ENV_REMOTE_STAGE_NAMES, "").strip()
    if not remote_names_raw:
        return []
    if not self_name:
        raise RuntimeError(
            f"{_ENV_REMOTE_STAGE_NAMES} is set but {_ENV_SELF_NAME} is not - "
            "TransportExecutor needs both to name its side of each RPC link"
        )

    remote_names = [n.strip() for n in remote_names_raw.split(",") if n.strip()]
    remote_hosts_raw = os.environ.get(_ENV_REMOTE_STAGE_HOSTS, "").strip()
    remote_hosts = [h.strip() for h in remote_hosts_raw.split(",") if h.strip()]
    if remote_hosts and len(remote_hosts) != len(remote_names):
        raise RuntimeError(
            f"{_ENV_REMOTE_STAGE_HOSTS} ({len(remote_hosts)} entries) must "
            f"match {_ENV_REMOTE_STAGE_NAMES} ({len(remote_names)} entries) "
            "1:1, or be left empty entirely (udp backend doesn't need it)"
        )

    rpc_port = int(os.environ.get(_ENV_RPC_PORT, str(_DEFAULT_RPC_PORT)))
    signaling_url = os.environ.get(_ENV_SIGNALING_URL) or None
    connect_timeout = float(os.environ.get(_ENV_CONNECT_TIMEOUT, "120"))

    from vllm.transport import TransportConfig, get_transport

    links: list[_RemoteStageLink] = []
    for i, remote_name in enumerate(remote_names):
        self_id = f"{self_name}-rpc-to-{remote_name}"
        peer_id = f"{remote_name}-rpc-to-{self_name}"
        host = remote_hosts[i] if remote_hosts else "127.0.0.1"
        # Real bug hit running this for real with 2+ remote stages: every
        # link here used the SAME local `rpc_port` to bind its own UDP
        # socket. SO_REUSEADDR (set in udp_transport.py's _connect_async)
        # lets all of them bind 0.0.0.0:rpc_port without error, but the
        # kernel then delivers each inbound datagram on that port to only
        # ONE of the driver's own sockets - not per-flow. With 2 remote
        # stages, packets actually from stage B would land on stage A's
        # protocol object (and vice versa) roughly at random, which looked
        # exactly like "the peer's NAT address keeps flapping between two
        # IPs" (logged as repeated NAT REBINDING DETECTED) - it was really
        # two different real peers' traffic being funneled into one socket.
        # Each of the driver's OWN outbound links needs a distinct local
        # port; the remote stage_server's own listen port (also `rpc_port`
        # by default) is unaffected since it only ever accepts the one
        # driver connection.
        local_rpc_port = rpc_port + i
        logger.info(
            "TransportExecutor: connecting RPC control channel to remote "
            "stage %s (%s:%s, local port %s)",
            remote_name,
            host,
            rpc_port,
            local_rpc_port,
        )
        transport = get_transport()
        if envs.VLLM_TRANSPORT == "quic-shared":
            # Same shared-QUIC-connection design as the PP tensor links
            # (see pipeline_bootstrap.py's establish_pp_transports) - this
            # RPC control channel is just one more named channel on the
            # SAME per-peer broker connection, not a separate
            # hole-punch/handshake of its own. `broker_socket_path` is
            # shared unmodified between the aioquic and Rust brokers - see
            # quic_rs_broker.py's module docstring.
            from vllm.transport.quic_broker import broker_socket_path

            transport.connect(
                TransportConfig(
                    self_id=self_id,
                    peer_id=peer_id,
                    connect_timeout=connect_timeout,
                    extra={
                        "broker_socket_path": broker_socket_path(self_name, remote_name),
                        "channel": "rpc",
                    },
                )
            )
        else:
            transport.connect(
                TransportConfig(
                    self_id=self_id,
                    peer_id=peer_id,
                    host=host,
                    port=rpc_port,
                    listen=False,  # driver always connects; stage_server always listens
                    signaling_url=signaling_url or "http://127.0.0.1:8000",
                    udp_mode="preserve",
                    udp_port=local_rpc_port,
                    connect_timeout=connect_timeout,
                )
            )
        logger.info("TransportExecutor: RPC control channel to %s connected", remote_name)
        link = _RemoteStageLink(name=remote_name, transport=transport, timeout=connect_timeout)
        if pipelined:
            link.pipelined = True
            link.reader_thread = threading.Thread(
                target=_reader_loop, args=(link,),
                name=f"transport-rpc-reader-{remote_name}", daemon=True,
            )
            link.reader_thread.start()
        links.append(link)
    return links


class TransportExecutor(MultiprocExecutor):
    """Driver-side executor for a real multi-machine PP deployment. See
    module docstring. Falls back to being a plain, unmodified
    `MultiprocExecutor` (no remote forwarding) if
    `VLLM_TRANSPORT_REMOTE_STAGE_NAMES` is unset.
    """

    def _init_executor(self) -> None:
        # This call spawns this machine's own local TP workers (they connect
        # torch.distributed/initialize_model_parallel() with the REAL,
        # already-1 pipeline_parallel_size their launch command passed in -
        # see stage_server.py/launch_pp_stage.py's `--pipeline-parallel-size 1`).
        # The vllm_config mutation below (batch-queue depth trick) happens
        # AFTER this returns specifically so those already-spawned, separate
        # OS processes never see it - see module docstring.
        super()._init_executor()
        pipelined = os.environ.get(_ENV_ENABLE_PIPELINING) == "1"
        self._remote_links: list[_RemoteStageLink] = _connect_remote_stages(pipelined=pipelined)
        self._fusion_enabled = pipelined and os.environ.get(_ENV_ENABLE_FUSION) == "1"
        if os.environ.get(_ENV_ENABLE_FUSION) == "1" and not pipelined:
            raise RuntimeError(
                f"{_ENV_ENABLE_FUSION} is set but {_ENV_ENABLE_PIPELINING} is not - "
                "fusion reuses the pipelined request-id-correlated dispatch path, "
                "it cannot run against the old lock-serialized one."
            )
        # request_id-agnostic here (each is a fresh id from the link's own
        # id_counter, assigned when the fused RPC actually goes out in
        # sample_tokens() below) - this only tracks the deferred
        # scheduler_output itself, one deque per link, FIFO. Safe without
        # a lock for push/pop ordering: execute_model()/sample_tokens() are
        # always called from EngineCore's single busy-loop thread, never
        # concurrently with each other - only the deque's use.
        self._pending_fusion: dict[str, "collections.deque"] = (
            {link.name: collections.deque() for link in self._remote_links}
            if self._fusion_enabled
            else {}
        )

        batch_queue_size_raw = os.environ.get(_ENV_BATCH_QUEUE_SIZE)
        if batch_queue_size_raw:
            batch_queue_size = int(batch_queue_size_raw)
            if not pipelined:
                raise RuntimeError(
                    f"{_ENV_BATCH_QUEUE_SIZE} is set but {_ENV_ENABLE_PIPELINING} "
                    "is not - setting a fake pipeline_parallel_size without also "
                    "switching to the request-id-correlated RPC dispatch just "
                    "reintroduces the crossed-response race step_with_batch_queue "
                    "would trigger against the old lock-serialized path. Both or "
                    "neither."
                )
            if batch_queue_size > 1:
                logger.info(
                    "TransportExecutor: mutating pipeline_parallel_size 1 -> %d "
                    "(driver-process-local only, workers already spawned above) "
                    "so vllm_config.max_concurrent_batches enables "
                    "step_with_batch_queue - see module docstring.",
                    batch_queue_size,
                )
                self.vllm_config.parallel_config.pipeline_parallel_size = batch_queue_size
        self._rpc_pool: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(
                max_workers=max(1, len(self._remote_links)),
                thread_name_prefix="transport-rpc",
            )
            if self._remote_links
            else None
        )
        # Separate pool for `_combine` (see `_forward_and_local`): `_combine`
        # itself just blocks on the `_dispatch_remote` futures it depends on
        # (plus the local future), it doesn't touch the network. Submitting
        # it to the SAME bounded `_rpc_pool` it depends on is a
        # self-starvation hazard once vLLM's own PP batch-queue pipelining
        # (`step_with_batch_queue`) has multiple steps in flight at once:
        # with `max_workers == len(remote_links)`, a `_combine` task can
        # occupy the last free worker while the `_dispatch_remote` task(s)
        # it's waiting on are still sitting in the queue behind it, unable
        # to ever get a worker - a real deadlock, not just slowness. Sized
        # generously since these threads are cheap (blocked, not computing).
        self._combine_pool: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(
                max_workers=max(4, len(self._remote_links) * 4),
                thread_name_prefix="transport-combine",
            )
            if self._remote_links
            else None
        )
        # 2026-08-15 microbatching fix: serializes local_future.result() calls
        # across overlapping steps' _combine tasks - see _forward_and_local's
        # matching comment for the real native crash this closes.
        self._local_wait_lock = threading.Lock()

    def _dispatch_remote(self, link: _RemoteStageLink, method: str, args: tuple, kwargs: dict):
        if link.pipelined:
            # Only reached via execute_dummy_batch (rare, not part of the
            # overlapping-steps hot path _forward_and_local handles
            # specially below) - fine to block synchronously start-to-finish
            # here, same net effect as the non-pipelined branch below.
            reply_future = self._dispatch_remote_pipelined_start(link, method, args, kwargs)
            return self._resolve_pipelined(link, method, reply_future)
        debug_timing = os.environ.get(_ENV_DEBUG_TIMING)
        t0 = time.perf_counter() if debug_timing else 0.0
        payload = cloudpickle.dumps((method, args, kwargs), protocol=pickle.HIGHEST_PROTOCOL)
        t1 = time.perf_counter() if debug_timing else 0.0
        # Serialized per-link: see _RemoteStageLink.lock docstring. Must hold
        # the lock across the FULL send+recv round trip, not just send() -
        # otherwise a second overlapping step's request could still be sent
        # (and its reply could still be pulled) while this one is still
        # waiting, which is exactly the crossed-response race this closes.
        with link.lock:
            link.transport.send(payload)
            t2 = time.perf_counter() if debug_timing else 0.0
            raw = link.transport.recv(timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS)
        t3 = time.perf_counter() if debug_timing else 0.0
        status, result = pickle.loads(raw)
        if debug_timing:
            print(
                f"[rpc-timing] {link.name} {method}: pickle={1000*(t1-t0):.1f}ms "
                f"send={1000*(t2-t1):.1f}ms wait_recv={1000*(t3-t2):.1f}ms "
                f"payload_bytes={len(payload)}",
                flush=True,
            )
        if status != _STATUS_OK:
            raise RuntimeError(f"remote stage {link.name!r} failed method {method!r}: {result}")
        return result

    def _dispatch_remote_pipelined_start(
        self, link: _RemoteStageLink, method: str, args: tuple, kwargs: dict
    ) -> "Future[tuple[str, object]]":
        """Pipelining-mode dispatch, SEND HALF ONLY: registers a pending
        Future keyed by a fresh request_id, sends, and returns immediately -
        does NOT block waiting for the reply. Must never be submitted to
        `_rpc_pool` and blocked-on there: that pool has exactly
        `len(remote_links)` workers (correct for the non-pipelined path,
        where at most one request per link is ever in flight by design -
        see `_dispatch_remote`'s lock). If this function DID block a pool
        worker for the full round trip, a second overlapping step's
        dispatch to the same link would have no worker to run on until the
        first step's reply arrives - silently reserializing everything
        `VLLM_TRANSPORT_ENABLE_PIPELINING` was supposed to remove. The
        actual wait happens in `_forward_and_local`'s `_combine`, which runs
        in `_combine_pool` (sized for exactly this: multiple steps' worth of
        concurrent waits, see that pool's own docstring).
        """
        request_id = next(link.id_counter)
        payload = cloudpickle.dumps(
            (request_id, method, args, kwargs), protocol=pickle.HIGHEST_PROTOCOL
        )
        reply_future: "Future[tuple[str, object]]" = Future()
        # 2026-08-15 hop-time profiling: the old [rpc-timing] instrumentation
        # in _dispatch_remote (pickle/send/wait_recv split) only runs on the
        # non-pipelined path - dead code whenever --enable-pipelining is on,
        # which is every real deployment that cares about this. Stash the
        # send timestamp + payload size directly on the Future (plain attr
        # assignment - concurrent.futures.Future has no __slots__, so this
        # is safe) so _resolve_pipelined can log a real round-trip number
        # for the pipelined path too, from whichever thread ends up waiting
        # on it (_combine, running on _combine_pool - never this thread).
        if os.environ.get(_ENV_DEBUG_TIMING):
            reply_future._debug_t_send = time.perf_counter()  # type: ignore[attr-defined]
            reply_future._debug_payload_bytes = len(payload)  # type: ignore[attr-defined]
        with link.pending_lock:
            link.pending[request_id] = reply_future
        try:
            link.transport.send(payload)
        except Exception:
            with link.pending_lock:
                link.pending.pop(request_id, None)
            raise
        return reply_future

    def _resolve_pipelined(
        self, link: _RemoteStageLink, method: str, reply_future: "Future[tuple[str, object]]"
    ):
        status, result = reply_future.result(timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS)
        t_send = getattr(reply_future, "_debug_t_send", None)
        if t_send is not None:
            round_trip_ms = 1000 * (time.perf_counter() - t_send)
            payload_bytes = getattr(reply_future, "_debug_payload_bytes", -1)
            print(
                f"[rpc-pipelined-timing] {link.name} {method}: "
                f"round_trip={round_trip_ms:.1f}ms payload_bytes={payload_bytes}",
                flush=True,
            )
        if status != _STATUS_OK:
            raise RuntimeError(f"remote stage {link.name!r} failed method {method!r}: {result}")
        return result

    def _forward_and_local(self, method: str, args: tuple, local_call, non_block: bool):
        """Fan out `method(*args)` to every remote stage while `local_call()`
        (the real, unmodified local MultiprocExecutor dispatch, always
        invoked non-blocking) runs concurrently, then wait for both. Only
        the LOCAL result is returned - matches the fact that only this
        machine's local workers ever produce a real `ModelRunnerOutput` (a
        non-last-PP-rank worker returns None, see `gpu_worker.py`'s
        `execute_model`), so the driver's own result is always the
        meaningful one.

        Non-pipelined links: dispatch (send+recv, the full round trip) runs
        as a blocking call on `_rpc_pool` (one worker per link, matching
        "at most one request in flight per link" - see `_dispatch_remote`).
        Pipelined links: only the SEND is done here (fast, not pool-bound);
        the reply wait happens inside `_combine` below, on `_combine_pool` -
        see `_dispatch_remote_pipelined_start`'s docstring for why that
        split matters for genuine multi-step overlap.
        """
        if not self._remote_links:
            return local_call()

        pending: list[tuple[_RemoteStageLink, object, bool]] = []
        for link in self._remote_links:
            if link.pipelined:
                reply_future = self._dispatch_remote_pipelined_start(link, method, args, {})
                pending.append((link, reply_future, True))
            else:
                pool_future = self._rpc_pool.submit(self._dispatch_remote, link, method, args, {})
                pending.append((link, pool_future, False))
        local_future = local_call()  # local_call itself passes non_block=True

        def _combine():
            debug_timing = os.environ.get(_ENV_DEBUG_TIMING)
            _t0 = time.perf_counter() if debug_timing else 0.0
            for link, fut, pipelined in pending:
                try:
                    if pipelined:
                        self._resolve_pipelined(link, method, fut)
                    else:
                        fut.result(timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS)
                except Exception:
                    logger.error("remote stage %s failed during %s", link.name, method)
                    raise
            if debug_timing:
                _t_remote_done = time.perf_counter()
            # 2026-08-15 microbatching fix: real crash hit running this for real
            # ("Bad address (src/fq.cpp:56)", a native libzmq fault - see
            # vllm/distributed/device_communicators/shm_broadcast.py, which
            # backs the LOCAL MultiprocExecutor's response path with a ZMQ
            # XPUB/SpinCondition pair). Root cause: with multiple steps'
            # `_combine` tasks now genuinely running concurrently on
            # `_combine_pool` (the whole point of pipelining the REMOTE side),
            # each one's `local_future.result()` call was ALSO happening
            # concurrently, from different threads, on the SAME underlying
            # local MultiprocExecutor response machinery. Real vLLM
            # `step_with_batch_queue()` never does this even with PP - it
            # always resolves batch_queue futures one at a time, sequentially,
            # from the single `run_busy_loop` thread; nothing in
            # shm_broadcast.py's ZMQ sockets was ever built or tested to be
            # waited on from multiple threads at once. This lock restores
            # that "one waiter at a time" invariant for the LOCAL leg only -
            # remote resolution above (this machine's own pipelined RPC code,
            # not vLLM's) stays fully concurrent across steps, since that
            # part was actually designed for it.
            with self._local_wait_lock:
                result = local_future.result()
            if debug_timing:
                _t_local_done = time.perf_counter()
                self_name = os.environ.get(_ENV_SELF_NAME, "?")
                print(
                    f"[COMBINE_TIMING] self={self_name} method={method} "
                    f"remote_wait_ms={1000*(_t_remote_done-_t0):.2f} "
                    f"local_wait_ms={1000*(_t_local_done-_t_remote_done):.2f} "
                    f"total_ms={1000*(_t_local_done-_t0):.2f}",
                    flush=True,
                )
            return result

        combined: Future = self._combine_pool.submit(_combine)
        return combined if non_block else combined.result()

    def execute_model(self, scheduler_output, non_block: bool = False):
        # 2026-08-15 microbatching perf fix: real regression measured running
        # this for real (single request, batch_queue_size=3: ~58-70% SLOWER
        # decode wall-clock than the old 1:1 step() baseline, not faster).
        # Root cause: step_with_batch_queue() keeps calling schedule() +
        # dispatching execute_model in a tight loop (no wait) until
        # len(batch_queue) == batch_queue_size, and the scheduler "may
        # return an empty batch if all requests are scheduled" (its own
        # docstring) - with only 1 real request in flight, 2 of every 3
        # dispatched steps have total_num_scheduled_tokens == 0, yet were
        # still getting a full RPC round trip to EVERY remote link for
        # nothing. Skip the remote fan-out entirely for a genuinely empty
        # schedule - local_call() below is equally a no-op for one, so
        # there is nothing remote stages need to be told about this step at
        # all (no PP tensor exchange happens for zero scheduled tokens
        # either). Only helps this specific waste; a real multi-request
        # workload that keeps the batch queue genuinely full end-to-end
        # would rarely hit this branch.
        if getattr(scheduler_output, "total_num_scheduled_tokens", 1) == 0:
            return super(TransportExecutor, self).execute_model(scheduler_output, non_block=non_block)
        if self._fusion_enabled:
            # Defer the remote half entirely - sample_tokens() below sends
            # ONE fused request per link instead of this method sending one
            # and sample_tokens() sending a second. Local dispatch is
            # unaffected (not part of remote RPC at all).
            for link in self._remote_links:
                self._pending_fusion[link.name].append(scheduler_output)
            return super(TransportExecutor, self).execute_model(scheduler_output, non_block=non_block)
        return self._forward_and_local(
            "execute_model",
            (scheduler_output,),
            lambda: super(TransportExecutor, self).execute_model(scheduler_output, non_block=True),
            non_block,
        )

    def sample_tokens(self, grammar_output, non_block: bool = False):
        if self._fusion_enabled:
            return self._forward_and_local_fused(grammar_output, non_block)
        return self._forward_and_local(
            "sample_tokens",
            (grammar_output,),
            lambda: super(TransportExecutor, self).sample_tokens(grammar_output, non_block=True),
            non_block,
        )

    def _forward_and_local_fused(self, grammar_output, non_block: bool):
        """RPC-fusion dispatch: pairs each remote link's deferred
        `execute_model` scheduler_output (stashed by `execute_model()`
        above) with this `sample_tokens` call's `grammar_output`, sending
        BOTH as one `execute_model_and_sample_tokens` request per link -
        see stage_server.py's matching handler. Falls back to a plain
        sample_tokens-only dispatch for a link with nothing queued (e.g.
        its paired execute_model hit the empty-batch fast path above and
        never deferred anything)."""
        if not self._remote_links:
            return super(TransportExecutor, self).sample_tokens(grammar_output, non_block=non_block)

        pending: list[tuple[_RemoteStageLink, "Future[tuple[str, object]]", str]] = []
        for link in self._remote_links:
            queue = self._pending_fusion[link.name]
            if queue:
                scheduler_output = queue.popleft()
                method = "execute_model_and_sample_tokens"
                reply_future = self._dispatch_remote_pipelined_start(
                    link, method, (scheduler_output, grammar_output), {}
                )
            else:
                method = "sample_tokens"
                reply_future = self._dispatch_remote_pipelined_start(
                    link, method, (grammar_output,), {}
                )
            pending.append((link, reply_future, method))
        local_future = super(TransportExecutor, self).sample_tokens(grammar_output, non_block=True)

        def _combine():
            debug_timing = os.environ.get(_ENV_DEBUG_TIMING)
            _t0 = time.perf_counter() if debug_timing else 0.0
            for link, fut, method in pending:
                try:
                    self._resolve_pipelined(link, method, fut)
                except Exception:
                    logger.error("remote stage %s failed during %s", link.name, method)
                    raise
            if debug_timing:
                _t_remote_done = time.perf_counter()
            with self._local_wait_lock:
                result = local_future.result()
            if debug_timing:
                _t_local_done = time.perf_counter()
                self_name = os.environ.get(_ENV_SELF_NAME, "?")
                print(
                    f"[COMBINE_TIMING] self={self_name} method=fused "
                    f"remote_wait_ms={1000*(_t_remote_done-_t0):.2f} "
                    f"local_wait_ms={1000*(_t_local_done-_t_remote_done):.2f} "
                    f"total_ms={1000*(_t_local_done-_t0):.2f}",
                    flush=True,
                )
            return result

        combined: Future = self._combine_pool.submit(_combine)
        return combined if non_block else combined.result()

    def execute_dummy_batch(self) -> None:
        if not self._remote_links:
            return super().execute_dummy_batch()
        remote_futures = [
            self._rpc_pool.submit(self._dispatch_remote, link, "execute_dummy_batch", (), {})
            for link in self._remote_links
        ]
        super().execute_dummy_batch()
        for f in remote_futures:
            f.result(timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS)

    def take_draft_token_ids(self):
        # Draft tokens are only produced/consumed on the driver's own
        # sampling side (spec decode); not forwarded.
        return super().take_draft_token_ids()

    def shutdown(self) -> None:
        for link in getattr(self, "_remote_links", []):
            if link.pipelined:
                link.reader_stop.set()
            try:
                link.transport.close()
            except Exception:
                logger.warning("error closing RPC link to %s during shutdown", link.name)
            if link.reader_thread is not None:
                link.reader_thread.join(timeout=5.0)
        rpc_pool = getattr(self, "_rpc_pool", None)
        if rpc_pool is not None:
            rpc_pool.shutdown(wait=False)
        combine_pool = getattr(self, "_combine_pool", None)
        if combine_pool is not None:
            combine_pool.shutdown(wait=False)
        super().shutdown()
