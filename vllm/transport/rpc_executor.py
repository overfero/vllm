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
"""
from __future__ import annotations

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
    lock: threading.Lock = field(default_factory=threading.Lock)


def _connect_remote_stages() -> list[_RemoteStageLink]:
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
        links.append(_RemoteStageLink(name=remote_name, transport=transport, timeout=connect_timeout))
    return links


class TransportExecutor(MultiprocExecutor):
    """Driver-side executor for a real multi-machine PP deployment. See
    module docstring. Falls back to being a plain, unmodified
    `MultiprocExecutor` (no remote forwarding) if
    `VLLM_TRANSPORT_REMOTE_STAGE_NAMES` is unset.
    """

    def _init_executor(self) -> None:
        super()._init_executor()
        self._remote_links: list[_RemoteStageLink] = _connect_remote_stages()
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

    def _dispatch_remote(self, link: _RemoteStageLink, method: str, args: tuple, kwargs: dict):
        debug_timing = os.environ.get("VLLM_TRANSPORT_DEBUG_TIMING")
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

    def _forward_and_local(self, method: str, args: tuple, local_call, non_block: bool):
        """Fan out `method(*args)` to every remote stage (background
        threads) while `local_call()` (the real, unmodified local
        MultiprocExecutor dispatch, always invoked non-blocking) runs
        concurrently, then wait for both. Only the LOCAL result is
        returned - matches the fact that only this machine's local
        workers ever produce a real `ModelRunnerOutput` (a non-last-PP-
        rank worker returns None, see `gpu_worker.py`'s `execute_model`),
        so the driver's own result is always the meaningful one.
        """
        if not self._remote_links:
            return local_call()

        remote_futures = [
            self._rpc_pool.submit(self._dispatch_remote, link, method, args, {})
            for link in self._remote_links
        ]
        local_future = local_call()  # local_call itself passes non_block=True

        def _combine():
            for f, link in zip(remote_futures, self._remote_links):
                try:
                    f.result(timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS)
                except Exception:
                    logger.error("remote stage %s failed during %s", link.name, method)
                    raise
            return local_future.result()

        combined: Future = self._combine_pool.submit(_combine)
        return combined if non_block else combined.result()

    def execute_model(self, scheduler_output, non_block: bool = False):
        return self._forward_and_local(
            "execute_model",
            (scheduler_output,),
            lambda: super(TransportExecutor, self).execute_model(scheduler_output, non_block=True),
            non_block,
        )

    def sample_tokens(self, grammar_output, non_block: bool = False):
        return self._forward_and_local(
            "sample_tokens",
            (grammar_output,),
            lambda: super(TransportExecutor, self).sample_tokens(grammar_output, non_block=True),
            non_block,
        )

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
            try:
                link.transport.close()
            except Exception:
                logger.warning("error closing RPC link to %s during shutdown", link.name)
        rpc_pool = getattr(self, "_rpc_pool", None)
        if rpc_pool is not None:
            rpc_pool.shutdown(wait=False)
        combine_pool = getattr(self, "_combine_pool", None)
        if combine_pool is not None:
            combine_pool.shutdown(wait=False)
        super().shutdown()
