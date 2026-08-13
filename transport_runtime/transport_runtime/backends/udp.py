"""UDPBackend: adapts an external UDP hole-punch transport to the
transport_runtime `Backend` interface.

This module does NOT reimplement hole punching, STUN, NAT keepalive/rebind
detection, or the wire protocol - all of that is imported unmodified from
the external hole-punch library and reused as-is (ported from vLLM's
`vllm/transport/udp_transport.py`, itself proven across 3 real NAT'd
machines - see README_LIVE_DEPLOYMENT_LOG.md in the vllm checkout). The
only new code here, same as in the original, is:

  1. Locating and importing that existing module (`peer.py`) as a library.
  2. A thin `asyncio.DatagramProtocol` subclass adding one new wire tag
     ("M", for application Message chunk) that the original protocol does
     not recognize and therefore silently ignores - so this is strictly
     additive, not a modification of the original dispatch behavior.
  3. A thread that runs an asyncio event loop, so send()/recv()/close()
     can present the same blocking, synchronous-looking interface as
     TCPBackend.

Honesty note on the close/liveness contract (see `ConnectionClosedError`'s
docstring in `backend.py`): UDP has no FIN. This backend only raises
`ConnectionClosedError` for a *local* `close()` unblocking a pending
`recv()`. A peer that silently vanishes is indistinguishable from one that
is just slow, until a caller's own `recv(timeout=...)` gives up - that
surfaces as a plain `TimeoutError`, same as before extraction. Nothing
about extracting this into transport_runtime changes that physical
limitation; it is documented here so it stays a stated tradeoff, not an
implicit gap the way it was before this interface had an explicit
liveness contract at all.
"""
from __future__ import annotations

import asyncio
import contextlib
import itertools
import os
import queue
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Final

from transport_runtime.backend import Backend, ConnectionClosedError, ConnectParams


def _locate_existing_transport_dir() -> Path:
    """Find the directory containing the external hole-punch transport's
    peer.py. Configurable via TRANSPORT_RUNTIME_UDP_HOLEPUNCH_DIR (falls
    back to the vLLM-integration-era env var name for continuity with
    existing launch scripts) since this transport lives outside this
    package entirely - it predates and is independent of both vLLM and
    transport_runtime.
    """
    env_path = os.environ.get("TRANSPORT_RUNTIME_UDP_HOLEPUNCH_DIR") or os.environ.get(
        "VLLM_UDP_TRANSPORT_DIR"
    )
    candidates = [Path(env_path)] if env_path else []
    candidates += [
        Path(__file__).resolve().parents[3] / "udp_holepunch",
    ]
    for candidate in candidates:
        if (candidate / "peer.py").exists():
            return candidate
    raise RuntimeError(
        "Could not locate the external UDP hole-punch transport (peer.py). "
        "Set TRANSPORT_RUNTIME_UDP_HOLEPUNCH_DIR to the directory that contains it."
    )


_existing_transport_dir = _locate_existing_transport_dir()
if str(_existing_transport_dir) not in sys.path:
    sys.path.insert(0, str(_existing_transport_dir))

import peer as _hp  # noqa: E402  (unmodified - the external transport, treated as a library)

_MSG_HEADER = struct.Struct("!III")  # msg_id, total_len, offset
_MSG_TAG = b"M"  # application message chunk
_STATUS_QUERY_TAG = b"Q"  # "how many bytes have you received for msg_id?"
_STATUS_ANSWER_TAG = b"A"
_STATUS_HEADER = struct.Struct("!II")  # msg_id, received_bytes
_QUERY_HEADER = struct.Struct("!I")  # msg_id
# None of these tags are used by the original protocol - see module docstring.

_MAX_SEND_ATTEMPTS = 5  # 1 initial send + up to 4 resends per batch if chunks were lost
_STATUS_TIMEOUT_BASE_S = 0.2  # first attempt: generous for loopback/LAN RTT, not backlog-scale
_STATUS_TIMEOUT_BACKOFF = 2.5  # each retry waits longer, in case it's backlog, not loss
_BATCH_CHUNKS = 500  # ~600KB per batch at CHUNK_PAYLOAD=1200
# Large messages are confirmed incrementally in batches rather than one all-or-nothing
# round at the very end - see the original module's docstring (unchanged reasoning,
# ported verbatim) for why a flat timeout would otherwise scale wrong with message size.

_CLOSED: Final = object()  # sentinel: local close() unblocked a pending recv()


class _InboundMessage:
    __slots__ = ("buf", "total_len", "received_bytes", "seen_offsets", "completed")

    def __init__(self, total_len: int) -> None:
        self.buf = bytearray(total_len)
        self.total_len = total_len
        self.received_bytes = 0
        self.seen_offsets: set[int] = set()
        self.completed = False

    def add_chunk(self, offset: int, chunk: bytes) -> bool:
        """Returns True the moment the message becomes fully assembled."""
        if self.completed or offset in self.seen_offsets:
            return False
        self.seen_offsets.add(offset)
        self.buf[offset : offset + len(chunk)] = chunk
        self.received_bytes += len(chunk)
        if self.received_bytes >= self.total_len:
            self.completed = True
            return True
        return False


class _AdapterProtocol(_hp.PeerProtocol):
    """Adds application-message framing and reliability on top of the
    external protocol. See module docstring for the additive-only
    contract this preserves.
    """

    def __init__(self, peer_addr: tuple[str, int], recv_queue: "queue.Queue[bytes | object]") -> None:
        super().__init__(peer_addr)
        self._recv_queue = recv_queue
        self._inbound: dict[int, _InboundMessage] = {}
        self.status_waiters: dict[int, asyncio.Future] = {}

    def _handle_datagram(self, data: bytes, addr) -> None:
        super()._handle_datagram(data, addr)  # preserve all original behavior
        if not data:
            return
        tag = data[0:1]
        try:
            if tag == _MSG_TAG:
                msg_id, total_len, offset = _MSG_HEADER.unpack(data[1:13])
                chunk = data[13:]
                msg = self._inbound.get(msg_id)
                if msg is None:
                    msg = self._inbound[msg_id] = _InboundMessage(total_len)
                if msg.add_chunk(offset, chunk):
                    self._recv_queue.put_nowait(bytes(msg.buf))
            elif tag == _STATUS_QUERY_TAG:
                (msg_id,) = _QUERY_HEADER.unpack(data[1:5])
                msg = self._inbound.get(msg_id)
                received = msg.received_bytes if msg is not None else 0
                self.send(_STATUS_ANSWER_TAG + _STATUS_HEADER.pack(msg_id, received))
            elif tag == _STATUS_ANSWER_TAG:
                msg_id, received = _STATUS_HEADER.unpack(data[1:9])
                fut = self.status_waiters.get(msg_id)
                if fut and not fut.done():
                    fut.set_result(received)
        except struct.error:
            return

    async def query_status(self, msg_id: int, timeout: float) -> int | None:
        fut = self.loop.create_future()
        self.status_waiters[msg_id] = fut
        self.send(_STATUS_QUERY_TAG + _QUERY_HEADER.pack(msg_id))
        try:
            return await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            self.status_waiters.pop(msg_id, None)


class UDPBackend(Backend):
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._protocol: _AdapterProtocol | None = None
        self._transport_obj: asyncio.DatagramTransport | None = None
        self._recv_queue: "queue.Queue[bytes | object]" = queue.Queue()
        self._msg_ids = itertools.count(1)
        self._closed = threading.Event()

    def connect(self, params: ConnectParams) -> None:
        if params.udp is None:
            raise ValueError("UDPBackend.connect() requires params.udp to be set")
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        fut = asyncio.run_coroutine_threadsafe(self._connect_async(params), self._loop)
        fut.result(timeout=params.connect_timeout + 5)

    def _run_loop(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _connect_async(self, params: ConnectParams) -> None:
        config = params.udp
        assert config is not None
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        for opt in (socket.SO_RCVBUF, socket.SO_SNDBUF):
            with contextlib.suppress(OSError):
                sock.setsockopt(socket.SOL_SOCKET, opt, _hp.SOCKET_BUFFER_REQUEST)
        sock.bind(("0.0.0.0", config.port))
        granted_rcvbuf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        pacing_burst = max(16, min(4096, granted_rcvbuf // _hp.CHUNK_PAYLOAD // 4))

        if config.mode == "stun":
            own_ip, own_port = _hp.stun_get_mapped_address(sock, config.stun_host, config.stun_port)
        else:
            own_ip, own_port = None, sock.getsockname()[1]

        reg_resp = _hp.register(config.signaling_url, params.self_id, own_port)
        own_ip = own_ip or reg_resp["public_ip"]

        peer_info = _hp.wait_for_peer(config.signaling_url, params.self_id, params.peer_id)
        coordinator_peer_addr = (peer_info["public_ip"], peer_info["udp_port"])
        delay = peer_info["start_at"] - time.time()
        if delay > 0:
            await asyncio.sleep(delay)

        loop = asyncio.get_running_loop()
        transport, protocol = await loop.create_datagram_endpoint(
            lambda: _AdapterProtocol(coordinator_peer_addr, self._recv_queue), sock=sock
        )
        protocol.pacing_burst = pacing_burst

        punch_task = asyncio.create_task(_hp.punch_loop(protocol))
        deadline = time.monotonic() + params.connect_timeout
        while not protocol.established and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        punch_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await punch_task

        if not protocol.established:
            transport.close()
            raise ConnectionError(
                f"UDP hole punch failed: no packet from peer within {params.connect_timeout}s"
            )

        self._transport_obj = transport
        self._protocol = protocol

    def send(self, data: bytes) -> None:
        if self._closed.is_set():
            raise ConnectionClosedError("send() called after close()")
        if self._loop is None or self._protocol is None:
            raise RuntimeError("send() called before connect()")
        fut = asyncio.run_coroutine_threadsafe(self._send_async(data), self._loop)
        fut.result()

    async def _send_async(self, data: bytes) -> None:
        assert self._protocol is not None
        msg_id = next(self._msg_ids)
        total_len = len(data)
        chunk_size = _hp.CHUNK_PAYLOAD
        burst = getattr(self._protocol, "pacing_burst", 64)
        offsets = list(range(0, max(total_len, 1), chunk_size))

        async def _send_indices(indices: range) -> None:
            for j, idx in enumerate(indices):
                offset = offsets[idx]
                chunk = data[offset : offset + chunk_size]
                header = _MSG_TAG + _MSG_HEADER.pack(msg_id, total_len, offset)
                self._protocol.send(header + chunk)
                if j % burst == burst - 1:
                    # Real sleep, not sleep(0): gives the receiver wall-clock time to
                    # drain its socket buffer instead of overflowing it.
                    await asyncio.sleep(_hp.PACING_GAP_SECONDS)

        for batch_start in range(0, len(offsets), _BATCH_CHUNKS):
            batch_end = min(batch_start + _BATCH_CHUNKS, len(offsets))
            batch_indices = range(batch_start, batch_end)
            last_offset = offsets[batch_end - 1]
            target_bytes = last_offset + min(chunk_size, total_len - last_offset)

            await _send_indices(batch_indices)
            for attempt in range(_MAX_SEND_ATTEMPTS):
                timeout = _STATUS_TIMEOUT_BASE_S * (_STATUS_TIMEOUT_BACKOFF**attempt)
                received = await self._protocol.query_status(msg_id, timeout)
                if received is not None and received >= target_bytes:
                    break
                if attempt == _MAX_SEND_ATTEMPTS - 1:
                    raise ConnectionError(
                        f"message {msg_id}: batch ending at byte {target_bytes}/{total_len} "
                        f"still incomplete after {_MAX_SEND_ATTEMPTS} attempts - receiver "
                        f"confirmed {received if received is not None else 'no response'} bytes"
                    )
                await _send_indices(batch_indices)

    def recv(self, timeout: float | None = None) -> bytes:
        try:
            item = self._recv_queue.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"recv() timed out after {timeout}s") from None
        if item is _CLOSED:
            self._recv_queue.put(_CLOSED)  # let other waiters see it too
            raise ConnectionClosedError("UDP backend was closed locally")
        return item  # type: ignore[return-value]

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._recv_queue.put(_CLOSED)
        if self._loop is not None:
            if self._transport_obj is not None:
                self._loop.call_soon_threadsafe(self._transport_obj.close)
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
