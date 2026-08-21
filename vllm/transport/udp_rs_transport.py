"""RawUdpRsTransport: replaces `udp_transport.py`'s reliability/chunking
layer with `vllm._rust_udp_raw_engine`'s native `send_message`/
`recv_message` (Rust `sendmmsg(2)`/`recvmmsg(2)` batching, Go-Back-N
retransmission, ack-loss recovery, per-message `msg_id` disambiguation -
see `rust/src/udp_raw_engine/src/lib.rs`'s module docstring and
[[project_raw_udp_rs_production_readiness]] for the three real bugs found
and fixed getting it to this point). Registered as this project's `"udp"`
backend - see `factory.py`.

Hole punching, STUN, and NAT keepalive are reused UNMODIFIED from the
existing standalone transport (`peer.py`, imported as `_hp`) - exactly
the same machinery `udp_transport.py` used, and exactly the same
rationale (proven, not worth reimplementing). What's different from
`udp_transport.py`:

  1. `peer.py`'s `PeerProtocol` (via asyncio) drives ONLY the hole-punch
     handshake phase. Once `protocol.established` is True, the real
     socket (the same `socket.socket` object handed to
     `loop.create_datagram_endpoint`, kept directly rather than fetched
     through asyncio's restricted `TransportSocket` wrapper) is
     `connect()`-ed to the peer's now-known address, and
     `PyRawUdpEngine.from_fd()` takes an independent `dup()`-ed copy of
     its fd BEFORE the asyncio transport is closed - so the Rust engine
     keeps working with a live, connected socket after asyncio and its
     event-loop thread are torn down entirely. All subsequent data-path
     I/O (`send_message`/`recv_message`) runs natively in Rust with the
     GIL released, not through asyncio's one-packet-per-callback loop.

  2. NAT keepalive during idle periods (the same real problem
     `udp_transport.py`'s own keepalive loop was built for - a NAT
     mapping/pinhole silently expiring after ~4 minutes of no traffic)
     is now a plain background Python thread calling `send_batch` with a
     tiny 1-byte ping, not an asyncio task - there's no event loop left
     to schedule one on after the handoff. The ping's tag byte
     (`_PING_TAG`) is neither `TYPE_DATA` (0) nor `TYPE_ACK` (1), so the
     engine's own `poll_data`/`poll_acks` silently ignore it on arrival
     (see the Rust crate's `poll_acks`/`poll_data` docstrings) - no reply
     is needed for its purpose (refreshing the LOCAL outbound NAT
     mapping), matching the old transport's own ping/pong being a
     one-way keepalive in practice.

  3. Multi-address NAT-rebind resilience (the old `_AdapterProtocol`'s
     `_recent_peer_addrs` list, sending application data to every
     recently-seen address to survive one side's NAT round-robining
     across external IPs) is a KNOWN GAP here, not carried over - the
     hole-punch phase still detects and logs a rebind via `peer.py`'s own
     `_handle_datagram`, but once handed off to the Rust engine, `connect()`
     locks the socket to a single peer address for its entire remaining
     lifetime. Acceptable for now since the raw UDP engine's own
     reliability (retransmission, ack-loss recovery) was the higher-value
     gap to close first; revisit if rebind-during-data-phase turns out to
     matter in practice on the real target deployment.
"""
from __future__ import annotations

import asyncio
import contextlib
import socket
import threading
import time
from pathlib import Path

from vllm.logger import init_logger
from vllm.transport.base import Transport, TransportConfig

logger = init_logger(__name__)

try:
    import vllm._rust_udp_raw_engine as _ue
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "RawUdpRsTransport requires the compiled vllm._rust_udp_raw_engine "
        "extension (built from rust/src/udp_raw_engine/python via "
        "./build_rust.sh) - it was not found. Original error: "
        f"{_exc}"
    ) from _exc


def _locate_existing_transport_dir() -> Path:
    import os
    import sys

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
import sys  # noqa: E402

if str(_existing_transport_dir) not in sys.path:
    sys.path.insert(0, str(_existing_transport_dir))

import peer as _hp  # noqa: E402  (unmodified - hole-punch/STUN only, see module docstring)

_CHUNK_PAYLOAD = 61440 - 13  # loopback-MTU-sized default; MSG_DATA_HEADER_BYTES=13 - see the Rust crate
_KEEPALIVE_INTERVAL_SECONDS = 15.0  # same interval/rationale as the old UDPTransport's keepalive
_PING_TAG = b"\xff"  # neither TYPE_DATA(0) nor TYPE_ACK(1) - silently ignored by the peer's engine
_DEFAULT_SEND_TIMEOUT = 60.0  # `Transport.send()` has no timeout parameter - see base.py
_NO_TIMEOUT_MS = 2**31 - 1  # `recv(timeout=None)` - Rust's recv_message needs a real (very large) deadline


class RawUdpRsTransport(Transport):
    def __init__(self) -> None:
        self._engine: "_ue.PyRawUdpEngine | None" = None
        self._sock: socket.socket | None = None
        self._chunk_payload = _CHUNK_PAYLOAD
        self._batch = 1
        self._window = 1
        self._keepalive_thread: threading.Thread | None = None
        self._keepalive_stop = threading.Event()
        self._self_id = "?"
        self._peer_id = "?"

    def connect(self, config: TransportConfig) -> None:
        self._self_id = config.self_id
        self._peer_id = config.peer_id
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        fut = asyncio.run_coroutine_threadsafe(self._connect_async(config), loop)
        try:
            fut.result(timeout=config.connect_timeout + 5)
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=5.0)

    async def _connect_async(self, config: TransportConfig) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        for opt in (socket.SO_RCVBUF, socket.SO_SNDBUF):
            with contextlib.suppress(OSError):
                sock.setsockopt(socket.SOL_SOCKET, opt, _hp.SOCKET_BUFFER_REQUEST)
        sock.bind(("0.0.0.0", config.udp_port))

        if config.udp_mode == "stun":
            own_ip, own_port = _hp.stun_get_mapped_address(sock, config.stun_host, config.stun_port)
        else:
            own_ip, own_port = None, sock.getsockname()[1]

        reg_resp = _hp.register(config.signaling_url, config.self_id, own_port)
        own_ip = own_ip or reg_resp["public_ip"]

        peer_info = _hp.wait_for_peer(config.signaling_url, config.self_id, config.peer_id)
        coordinator_peer_addr = (peer_info["public_ip"], peer_info["udp_port"])
        delay = peer_info["start_at"] - time.time()
        if delay > 0:
            await asyncio.sleep(delay)

        loop = asyncio.get_running_loop()
        transport, protocol = await loop.create_datagram_endpoint(
            lambda: _hp.PeerProtocol(coordinator_peer_addr), sock=sock
        )

        punch_task = asyncio.create_task(_hp.punch_loop(protocol))
        deadline = time.monotonic() + config.connect_timeout
        while not protocol.established and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        punch_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await punch_task

        if not protocol.established:
            transport.close()
            raise ConnectionError(
                f"UDP hole punch failed: no packet from peer within {config.connect_timeout}s"
            )

        # Hand off from asyncio to the Rust engine - see module docstring
        # point 1 for why `from_fd` (a real independent `dup()`) must
        # happen BEFORE `transport.close()`.
        sock.connect(protocol.peer_addr)
        self._engine = _ue.PyRawUdpEngine.from_fd(sock.fileno())
        sndbuf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
        self._batch = max(1, sndbuf // self._chunk_payload)
        self._window = self._batch
        self._sock = sock
        transport.close()

        self._keepalive_stop.clear()
        self._keepalive_thread = threading.Thread(target=self._keepalive_loop, daemon=True)
        self._keepalive_thread.start()

    def _keepalive_loop(self) -> None:
        while not self._keepalive_stop.wait(_KEEPALIVE_INTERVAL_SECONDS):
            assert self._engine is not None
            with contextlib.suppress(OSError):
                self._engine.send_batch([_PING_TAG])

    def send(self, data: bytes) -> None:
        if self._engine is None:
            raise RuntimeError("send() called before connect()")
        _t0 = time.monotonic()
        self._engine.send_message(
            data, self._chunk_payload, self._batch, self._window, int(_DEFAULT_SEND_TIMEOUT * 1000)
        )
        logger.info(
            "[TRANSPORT_TIMING] self=%s peer=%s op=send bytes=%d duration_ms=%.2f",
            self._self_id, self._peer_id, len(data), (time.monotonic() - _t0) * 1000,
        )

    def recv(self, timeout: float | None = None) -> bytes:
        if self._engine is None:
            raise RuntimeError("recv() called before connect()")
        timeout_ms = _NO_TIMEOUT_MS if timeout is None else max(1, int(timeout * 1000))
        _t0 = time.monotonic()
        try:
            data, _chunks, _gap = self._engine.recv_message(self._chunk_payload, self._batch, self._window, timeout_ms)
        except OSError as exc:
            raise TimeoutError(f"recv() timed out after {timeout}s") from exc
        logger.info(
            "[TRANSPORT_TIMING] self=%s peer=%s op=recv bytes=%d duration_ms=%.2f",
            self._self_id, self._peer_id, len(data), (time.monotonic() - _t0) * 1000,
        )
        return bytes(data)

    def close(self) -> None:
        self._keepalive_stop.set()
        if self._keepalive_thread is not None:
            self._keepalive_thread.join(timeout=5.0)
        self._engine = None  # drops the Rust engine's independent dup'd fd
        if self._sock is not None:
            self._sock.close()
            self._sock = None
