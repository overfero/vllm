"""Shared, backend-agnostic broker infrastructure: the local Unix-domain-
socket IPC hop that lets real client processes reach a named channel on
the shared QUIC connection without speaking QUIC themselves - see
`quic_broker.py`'s module docstring for the full design rationale.

Extracted into its own module (previously part of the aioquic-era
`quic_broker.py`, now deleted) once `quic_broker.py` became a single
Rust-native implementation with nothing aioquic-specific left in it -
none of the code here ever touched `QuicConnection`/`aioquic` at all, it
only calls a broker object's `_send_on_channel_async`/
`_recv_from_channel_async`/`_notify_local_peer_gone_async`/
`_start_local_server_async` methods and reads `broker._closed_exc`/
`broker._loop`/`broker._thread` - fully duck-typed against whatever
`QuicBroker` implementation is current.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import struct
import threading
from typing import Protocol

from vllm.logger import init_logger

logger = init_logger(__name__)

_LOCAL_LEN_PREFIX = struct.Struct("!Q")  # 8-byte length prefix, local-socket messages
_LOCAL_PEER_GONE = object()  # sentinel - see notify_local_peer_gone callers


class _BrokerLike(Protocol):
    _loop: asyncio.AbstractEventLoop | None
    _thread: threading.Thread | None
    _closed_exc: Exception | None

    async def _send_on_channel_async(self, channel: str, data: bytes) -> None: ...
    async def _recv_from_channel_async(self, channel: str, timeout: float | None) -> bytes: ...
    async def _notify_local_peer_gone_async(self, channel: str) -> None: ...
    async def _start_local_server_async(self, socket_path: str) -> None: ...


def broker_socket_path(self_id: str, peer_id: str) -> str:
    """Deterministic path so a client process (which only knows its own
    self_id/peer_id, not the broker's internal state) can find the right
    socket. Keyed by BOTH names, not just self_id, so a middle-of-chain
    PP stage (two distinct peers - the previous stage and the next one,
    each needing its own separate broker connection) gets two non-
    colliding sockets rather than one broker's channels silently
    overwriting the other's. Under /tmp (not /kaggle/working - this is
    ephemeral IPC plumbing, not output - see the project's own placement
    convention)."""
    def _safe(s: str) -> str:
        return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)

    return f"/tmp/vllm_quic_broker_{_safe(self_id)}__{_safe(peer_id)}.sock"


async def _handle_local_client(
    broker: _BrokerLike, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
) -> None:
    """Runs on `broker._loop` - the local Unix-socket listener's own
    asyncio loop (separate from whatever drives the real QUIC connection
    underneath `broker` - a dedicated Rust thread, for the current
    implementation - see `quic_broker.py`'s module docstring). Pumps
    bytes both directions between one local client connection and one
    named channel.

    Real bug hit building the ORIGINAL version of this function, kept
    here since the same shutdown hazard still applies: `_pump_from_quic`
    must never block forever on a message that will never arrive once the
    local peer has gone away. Two fix attempts that were themselves real
    bugs, in order: (1) cancelling `_pump_from_quic` the moment
    `_pump_to_quic` hit EOF doesn't stop an already-running blocking call
    in a thread pool - the in-flight call could still pull a REAL,
    not-yet-delivered message with nothing left listening for the result;
    (2) polling instead of blocking indefinitely avoided that, but added
    enough scheduling churn under several concurrent channels to make
    throughput worse across the board. The actual fix: `_pump_to_quic`,
    on EOF, pushes a sentinel through the SAME FIFO queue
    `_recv_from_channel_async` already reads from
    (`_notify_local_peer_gone_async`) - `_pump_from_quic`'s
    `await q.get()`-based wait wakes up naturally and immediately, in
    queue order (so every real message already queued ahead of the
    sentinel is still delivered first), no polling, no race.
    """
    channel = "?"
    try:
        name_line = await reader.readline()
        channel = name_line.decode("utf-8").strip()
        if not channel:
            writer.close()
            return

        async def _pump_to_quic() -> None:
            try:
                while True:
                    header = await reader.readexactly(_LOCAL_LEN_PREFIX.size)
                    (n,) = _LOCAL_LEN_PREFIX.unpack(header)
                    payload = await reader.readexactly(n)
                    if broker._closed_exc is not None:
                        raise broker._closed_exc
                    await broker._send_on_channel_async(channel, payload)
            finally:
                await broker._notify_local_peer_gone_async(channel)

        async def _pump_from_quic() -> None:
            while True:
                data = await broker._recv_from_channel_async(channel, None)
                if data is _LOCAL_PEER_GONE:
                    return
                writer.write(_LOCAL_LEN_PREFIX.pack(len(data)) + data)
                await writer.drain()

        results = await asyncio.gather(_pump_to_quic(), _pump_from_quic(), return_exceptions=True)
        for r in results:
            if isinstance(r, (asyncio.IncompleteReadError, ConnectionError)):
                continue  # expected: local peer closed its side, or the QUIC link died
            if isinstance(r, BaseException):
                logger.warning(
                    "QuicBroker local channel %r: pump task failed unexpectedly: %r", channel, r
                )
    finally:
        with contextlib.suppress(Exception):
            writer.close()


def serve_local_broker(broker: _BrokerLike, socket_path: str) -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
    """Starts the local Unix-socket listener that fans channels out to
    real client processes, on `broker._loop`/`broker._thread` (already
    set up by the preceding `QuicBroker.connect()` call - this function
    only adds the listener to that already-running loop, it doesn't start
    a new one). Blocks the calling thread until the local listener is
    actually accepting connections."""
    assert broker._loop is not None and broker._thread is not None, (
        "serve_local_broker() must be called after QuicBroker.connect() has returned"
    )
    fut = asyncio.run_coroutine_threadsafe(broker._start_local_server_async(socket_path), broker._loop)
    fut.result(timeout=10.0)
    return broker._loop, broker._thread


__all__ = [
    "broker_socket_path", "_handle_local_client", "serve_local_broker", "_LOCAL_PEER_GONE", "_LOCAL_LEN_PREFIX",
]
