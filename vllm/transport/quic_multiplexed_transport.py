"""QuicMultiplexedTransport: a `Transport` implementation for one named
channel of a shared `QuicBroker` connection (see `quic_broker.py` for the
full rationale and wire design). Drop-in anywhere `get_transport()` is
called - callers never see the broker or the local IPC hop underneath,
same external contract as `TCPTransport`/`UDPTransport`/`QUICTransport`.

This process talks to the broker (a separate, already-running process on
the SAME machine - see `quic_broker.py`'s `serve_local_broker`) over a
local Unix-domain socket, not QUIC directly: connect, send one line
identifying the wanted channel name, then exchange 8-byte-length-prefixed
messages exactly like every other byte-oriented boundary in this project's
transport stack. `TransportConfig.extra` carries the two broker-specific
fields this needs (`broker_socket_path`, `channel`) - see `base.py`'s
`extra: dict` field, which exists precisely so a backend-specific need
like this one doesn't force new top-level fields onto every other
backend's config.
"""
from __future__ import annotations

import socket
import struct
import time

from vllm.logger import init_logger
from vllm.transport.base import Transport, TransportConfig

logger = init_logger(__name__)

_LEN_PREFIX = struct.Struct("!Q")
_CONNECT_RETRY_INTERVAL_S = 0.2


class QuicMultiplexedTransport(Transport):
    def __init__(self) -> None:
        self._sock: socket.socket | None = None
        self._self_id = "?"
        self._peer_id = "?"
        self._recvbuf = bytearray()

    def connect(self, config: TransportConfig) -> None:
        self._self_id = config.self_id
        self._peer_id = config.peer_id
        socket_path = config.extra.get("broker_socket_path")
        channel = config.extra.get("channel")
        if not socket_path or not channel:
            raise ValueError(
                "QuicMultiplexedTransport requires config.extra={'broker_socket_path': ..., "
                "'channel': ...} - see quic_broker.py's broker_socket_path() for the "
                "expected path convention"
            )

        deadline = time.monotonic() + config.connect_timeout
        sock: socket.socket | None = None
        last_exc: Exception | None = None
        while time.monotonic() < deadline:
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(socket_path)
                break
            except (FileNotFoundError, ConnectionRefusedError) as exc:
                # Broker process may not have created its listening socket
                # yet (real, expected race at deployment startup - the
                # broker and this process are launched close together,
                # not strictly ordered) - retry until connect_timeout,
                # exactly like every other backend's own connect-retry
                # window, rather than failing on the very first attempt.
                last_exc = exc
                if sock is not None:
                    sock.close()
                time.sleep(_CONNECT_RETRY_INTERVAL_S)
        else:
            raise ConnectionError(
                f"QuicMultiplexedTransport: could not reach broker at {socket_path!r} "
                f"within {config.connect_timeout}s (is the broker process running?)"
            ) from last_exc

        assert sock is not None
        sock.sendall(channel.encode("utf-8") + b"\n")
        self._sock = sock
        logger.info(
            "QuicMultiplexedTransport: %s connected to broker channel %r (peer=%s)",
            self._self_id, channel, self._peer_id,
        )

    def send(self, data: bytes) -> None:
        if self._sock is None:
            raise RuntimeError("send() called before connect()")
        self._sock.sendall(_LEN_PREFIX.pack(len(data)) + data)

    def _recv_exact(self, n: int, deadline: float | None) -> bytes:
        assert self._sock is not None
        while len(self._recvbuf) < n:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("recv() timed out")
                self._sock.settimeout(remaining)
            else:
                self._sock.settimeout(None)
            try:
                chunk = self._sock.recv(max(65536, n - len(self._recvbuf)))
            except socket.timeout:
                raise TimeoutError("recv() timed out") from None
            if not chunk:
                raise ConnectionError("QuicMultiplexedTransport: broker connection closed")
            self._recvbuf += chunk
        out = bytes(self._recvbuf[:n])
        del self._recvbuf[:n]
        return out

    def recv(self, timeout: float | None = None) -> bytes:
        if self._sock is None:
            raise RuntimeError("recv() called before connect()")
        deadline = None if timeout is None else time.monotonic() + timeout
        header = self._recv_exact(_LEN_PREFIX.size, deadline)
        (n,) = _LEN_PREFIX.unpack(header)
        return self._recv_exact(n, deadline)

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None
