"""QuicBroker: one real QUIC connection per machine-pair, shared by
several logical channels (e.g. this project's per-TP-rank PP tensor
links plus the RPC control channel), instead of each channel hole-
punching and QUIC-handshaking its own separate connection - see the
module docstring history in git for the original design rationale (why
sharing one connection matters for this project's real 3-links-between-
the-same-2-machines topology), unchanged by this rewrite.

Rust-native end to end: `vllm._rust_quic_engine`'s
`PyMultiplexedConnectionDriver` owns the connection's whole lifetime
(handshake, timers, GSO send, per-channel stream framing, drain-before-
close) on a dedicated background thread - the multi-channel analogue of
`quic_transport.py`'s `QUICTransport` (see that module's docstring and
`rust/src/quic_engine/src/multiplexed_driver.rs`'s for the shared
architecture). Replaces BOTH of this project's earlier broker
implementations (the original aioquic-based `QuicBroker` and the Python-
asyncio-orchestrated `RustQuicBroker`) outright, consolidated into one
`"quic-shared"` name - same consolidation `"quic"`/`"quic-rs"` went
through.

Two pieces, unchanged in shape from the original design:

1. This module - `QuicBroker` drives the shared QUIC connection and
   exposes N independent named channels, each getting its own persistent
   unidirectional stream pair (see `multiplexed_driver.rs`'s wire-format
   docstring for the exact framing).
2. `quic_broker_common.py` (extracted, previously part of this file) - a
   tiny local Unix-domain-socket IPC hop so real client processes (this
   project's per-local-rank PP workers, the RPC control process) don't
   need to speak QUIC themselves at all. Fully backend-agnostic - never
   touched QUIC/aioquic/quinn-proto at all, just duck-types against
   whatever `QuicBroker` implementation is current (`_send_on_channel_async`/
   `_recv_from_channel_async`/`_notify_local_peer_gone_async`/
   `_start_local_server_async`/`_loop`/`_thread`/`_closed_exc`) - see that
   module's own docstring.

Since the real QUIC I/O now runs on a Rust thread (not cooperatively on
the SAME asyncio loop the way the old aioquic/Python-orchestrated designs
worked), `QuicBroker` still keeps ONE small asyncio loop of its own - not
for QUIC, just for hole-punch (short-lived, only during `connect()`) and
the local Unix-socket IPC server (`_start_local_server_async`, long-
lived). A dedicated `_dispatch_loop` background task bridges Rust events
into that loop: it repeatedly calls the driver's blocking `recv_any()`
(via `asyncio.to_thread`, since a raw blocking call would freeze the
whole loop) and fans each `(channel, data)` pair out into a per-channel
`asyncio.Queue`, mirroring the old design's `_deliver`/`_channel_queues`
mechanism exactly - `_recv_from_channel_async` just awaits its channel's
queue, same as before.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import sys
import threading
import time
from pathlib import Path

from vllm.logger import init_logger
from vllm.transport.quic_broker_common import (  # noqa: F401 (broker_socket_path/serve_local_broker re-exported)
    _LOCAL_PEER_GONE,
    _handle_local_client,
    broker_socket_path,
    serve_local_broker,
)

logger = init_logger(__name__)

try:
    import vllm._rust_quic_engine as _qe
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "QuicBroker requires the compiled vllm._rust_quic_engine "
        "extension (built from rust/src/quic_engine/python via "
        "./build_rust.sh) - it was not found. Original error: "
        f"{_exc}"
    ) from _exc


def _locate_existing_transport_dir() -> Path:
    env_path = os.environ.get("VLLM_UDP_TRANSPORT_DIR")
    candidates = [Path(env_path)] if env_path else []
    candidates += [Path(__file__).resolve().parents[2] / "udp_holepunch"]
    for candidate in candidates:
        if (candidate / "peer.py").exists():
            return candidate
    raise RuntimeError(
        "Could not locate the existing UDP hole-punch transport (peer.py). "
        "Set VLLM_UDP_TRANSPORT_DIR to the directory that contains it."
    )


_existing_transport_dir = _locate_existing_transport_dir()
if str(_existing_transport_dir) not in sys.path:
    sys.path.insert(0, str(_existing_transport_dir))

import peer as _hp  # noqa: E402  (unmodified - hole-punch/STUN/keepalive only)

DEFAULT_IDLE_TIMEOUT_S = float(os.environ.get("VLLM_TRANSPORT_QUIC_IDLE_TIMEOUT_S", "45"))
DEFAULT_MAX_MESSAGE_BYTES = int(
    os.environ.get("VLLM_TRANSPORT_QUIC_MAX_MESSAGE_BYTES", str(2 * 1024 * 1024 * 1024))
)
_FALLBACK_WINDOW_BYTES = 1024 * 1024
_KEEPALIVE_INTERVAL_SECONDS = 15.0
_DEFAULT_SEND_TIMEOUT_S = 60.0
_DRAIN_TIMEOUT_S = 3.0
_NO_TIMEOUT_MS = 2**31 - 1  # "block forever" - see recv_any()'s real-timeout limitation note below


class QuicBroker:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._driver: "_qe.PyMultiplexedConnectionDriver | None" = None
        self._sock: socket.socket | None = None
        self._keepalive_addr: tuple[str, int] | None = None
        self._keepalive_thread: threading.Thread | None = None
        self._keepalive_stop = threading.Event()
        self._closed_exc: Exception | None = None
        self._channel_queues: dict[str, asyncio.Queue] = {}
        self._dispatch_task: asyncio.Task | None = None
        self._local_server: asyncio.AbstractServer | None = None
        self._self_id = "?"
        self._peer_id = "?"

    def connect(
        self,
        *,
        self_id: str,
        peer_id: str,
        signaling_url: str,
        udp_port: int,
        listen: bool,
        connect_timeout: float = 120.0,
        idle_timeout: float = DEFAULT_IDLE_TIMEOUT_S,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
    ) -> None:
        self._self_id = self_id
        self._peer_id = peer_id
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        fut = asyncio.run_coroutine_threadsafe(
            self._connect_async(
                self_id=self_id, peer_id=peer_id, signaling_url=signaling_url, udp_port=udp_port,
                listen=listen, connect_timeout=connect_timeout, idle_timeout=idle_timeout,
                max_message_bytes=max_message_bytes,
            ),
            self._loop,
        )
        fut.result(timeout=connect_timeout + 5)

    def _run_loop(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _queue_for(self, channel: str) -> asyncio.Queue:
        q = self._channel_queues.get(channel)
        if q is None:
            q = self._channel_queues[channel] = asyncio.Queue()
            if self._closed_exc is not None:
                q.put_nowait(self._closed_exc)
        return q

    async def _connect_async(
        self, *, self_id: str, peer_id: str, signaling_url: str, udp_port: int, listen: bool,
        connect_timeout: float, idle_timeout: float, max_message_bytes: int,
    ) -> None:
        # ---- Phase 1: hole-punch, identical to quic_transport.py's own. ----
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        for opt in (socket.SO_RCVBUF, socket.SO_SNDBUF):
            with contextlib.suppress(OSError):
                sock.setsockopt(socket.SOL_SOCKET, opt, _hp.SOCKET_BUFFER_REQUEST)
        sock.bind(("0.0.0.0", udp_port))

        own_ip, own_port = None, sock.getsockname()[1]
        reg_resp = _hp.register(signaling_url, self_id, own_port)
        own_ip = own_ip or reg_resp["public_ip"]

        peer_info = _hp.wait_for_peer(signaling_url, self_id, peer_id)
        coordinator_peer_addr = (peer_info["public_ip"], peer_info["udp_port"])
        delay = peer_info["start_at"] - time.time()
        if delay > 0:
            await asyncio.sleep(delay)

        loop = asyncio.get_running_loop()
        transport, protocol = await loop.create_datagram_endpoint(
            lambda: _hp.PeerProtocol(coordinator_peer_addr), sock=sock
        )

        punch_task = asyncio.create_task(_hp.punch_loop(protocol))
        deadline = time.monotonic() + connect_timeout
        while not protocol.established and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        punch_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await punch_task

        if not protocol.established:
            transport.close()
            raise ConnectionError(f"QuicBroker: UDP hole punch failed within {connect_timeout}s")

        # ---- Phase 2: real QUIC handshake, driven entirely by the Rust
        # multi-channel driver - see quic_transport.py's _connect_async
        # for line-by-line rationale on the buffer-derived window sizing
        # below, identical here. ----
        idle_timeout_ms = max(1, int(idle_timeout * 1000))
        try:
            granted_rcvbuf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
            granted_sndbuf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
            window = max(64 * 1024, min(granted_rcvbuf, granted_sndbuf))
        except OSError:
            window = _FALLBACK_WINDOW_BYTES
        send_window = 8 * window
        receive_window = 8 * window
        max_congestion_window = 8 * window

        is_client = not listen
        handshake_timeout_ms = max(1, int(connect_timeout * 1000))
        try:
            if is_client:
                driver = _qe.PyMultiplexedConnectionDriver.connect_client(
                    sock.fileno(), coordinator_peer_addr[0], coordinator_peer_addr[1],
                    "vllm-pp-broker-transport", idle_timeout_ms, receive_window, send_window,
                    window, max_congestion_window, max_message_bytes, handshake_timeout_ms,
                )
            else:
                driver = _qe.PyMultiplexedConnectionDriver.connect_server(
                    sock.fileno(), idle_timeout_ms, receive_window, send_window,
                    window, max_congestion_window, max_message_bytes, handshake_timeout_ms,
                )
        except ValueError as exc:
            transport.close()
            raise ConnectionError(
                f"QuicBroker: QUIC handshake did not complete within {connect_timeout}s: {exc}"
            ) from None

        self._driver = driver
        self._sock = sock
        self._keepalive_addr = protocol.peer_addr
        transport.close()

        self._dispatch_task = asyncio.create_task(self._dispatch_loop())
        self._keepalive_stop.clear()
        self._keepalive_thread = threading.Thread(target=self._keepalive_loop, daemon=True)
        self._keepalive_thread.start()
        logger.info("QuicBroker: connection %s <-> %s established", self_id, peer_id)

    async def _dispatch_loop(self) -> None:
        """Bridges the Rust driver's blocking `recv_any()` into per-
        channel asyncio queues - see module docstring. Runs for the whole
        connection's lifetime; exits (after delivering the poison pill to
        every channel) once the connection closes."""
        assert self._driver is not None
        while True:
            try:
                channel, data = await asyncio.to_thread(self._driver.recv_any, _NO_TIMEOUT_MS)
            except ValueError as exc:
                self._closed_exc = ConnectionError(f"QuicBroker connection terminated: {exc}")
                for q in self._channel_queues.values():
                    q.put_nowait(self._closed_exc)
                return
            self._queue_for(channel).put_nowait(data)

    def _keepalive_loop(self) -> None:
        # NAT pinhole freshness only - see quic_transport.py's identical
        # keepalive for the same "no QUIC-level PING yet" note.
        seq = 0
        assert self._sock is not None and self._keepalive_addr is not None
        while not self._keepalive_stop.wait(_KEEPALIVE_INTERVAL_SECONDS):
            seq += 1
            with contextlib.suppress(OSError):
                self._sock.sendto(_hp.pack_ping(seq, time.monotonic()), self._keepalive_addr)

    # ---- async-native, called directly by quic_broker_common's
    # _handle_local_client on this SAME loop - the local-socket hop is on
    # this loop, the actual send/recv against Rust is one
    # `asyncio.to_thread` hop away (unavoidable now that real QUIC I/O
    # runs on a genuinely separate OS thread, not cooperatively on this
    # same loop the way the old aioquic/Python-orchestrated designs
    # allowed - see module docstring). ----

    async def _send_on_channel_async(self, channel: str, data: bytes) -> None:
        if self._closed_exc is not None:
            raise self._closed_exc
        assert self._driver is not None
        await asyncio.to_thread(
            self._driver.send_on_channel, channel, data, int(_DEFAULT_SEND_TIMEOUT_S * 1000)
        )

    async def _notify_local_peer_gone_async(self, channel: str) -> None:
        self._queue_for(channel).put_nowait(_LOCAL_PEER_GONE)

    async def _recv_from_channel_async(self, channel: str, timeout: float | None) -> bytes:
        q = self._queue_for(channel)
        try:
            data = await (q.get() if timeout is None else asyncio.wait_for(q.get(), timeout=timeout))
        except asyncio.TimeoutError:
            raise TimeoutError(f"recv_from_channel({channel!r}) timed out after {timeout}s") from None
        if isinstance(data, Exception):
            q.put_nowait(data)
            raise data
        return data

    async def _start_local_server_async(self, socket_path: str) -> None:
        with contextlib.suppress(FileNotFoundError):
            os.remove(socket_path)
        self._local_server = await asyncio.start_unix_server(
            lambda r, w: _handle_local_client(self, r, w), path=socket_path
        )

    def close(self) -> None:
        if self._loop is not None:
            fut = asyncio.run_coroutine_threadsafe(self._close_async(), self._loop)
            with contextlib.suppress(Exception):
                fut.result(timeout=11.0)
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    async def _close_async(self) -> None:
        self._keepalive_stop.set()
        if self._local_server is not None:
            self._local_server.close()
            with contextlib.suppress(Exception):
                await self._local_server.wait_closed()
        if self._driver is not None:
            # Close the Rust driver FIRST, then wait for _dispatch_task -
            # NOT the other way around. Real hazard avoided here:
            # cancelling an asyncio Task wrapping a still-running
            # `asyncio.to_thread` call does NOT stop the underlying
            # blocking call already executing in the thread pool (the
            # same well-known limitation quic_broker_common.py's
            # `_handle_local_client` docstring documents for its own
            # pump tasks) - `_dispatch_task` is blocked inside
            # `recv_any()` for up to `_NO_TIMEOUT_MS` at any given
            # moment, so cancelling it first would just leave this
            # coroutine awaiting a task that never actually stops until
            # the underlying call returns anyway. Closing the driver
            # first makes `recv_any()` return (with a real error, once
            # the driver thread's own shutdown sequence completes) on
            # its own, letting `_dispatch_task` finish naturally and
            # promptly.
            driver = self._driver
            self._driver = None
            await asyncio.to_thread(driver.close, int(_DRAIN_TIMEOUT_S * 1000))
        if self._dispatch_task is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._dispatch_task, timeout=5.0)
        if self._sock is not None:
            self._sock.close()
            self._sock = None


__all__ = ["QuicBroker", "broker_socket_path", "serve_local_broker"]
