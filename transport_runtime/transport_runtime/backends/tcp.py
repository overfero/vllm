"""Plain-TCP `Backend` implementation.

Deliberately independent of any framework's own rendezvous machinery
(e.g. vLLM's torch.distributed TCPStore usage) - this is a standalone
data-plane/control-plane backend for the transport_runtime abstraction,
not a wrapper around framework-specific plumbing. Ported from vLLM's
`vllm/transport/tcp_transport.py` (proven, unmodified logic) onto the new
`Backend`/`ConnectParams` interface - see `backend.py` for what changed
and why.
"""
from __future__ import annotations

import queue
import socket
import struct
import threading
import time
from typing import Final

from transport_runtime.backend import Backend, ConnectionClosedError, ConnectParams

_LEN_HEADER = struct.Struct("!Q")  # 8-byte big-endian message length prefix
_CLOSED: Final = object()  # sentinel: peer closed (or we did) - distinct from "nothing yet"


class TCPBackend(Backend):
    def __init__(self) -> None:
        self._sock: socket.socket | None = None
        self._listener: socket.socket | None = None
        self._recv_queue: "queue.Queue[bytes | object]" = queue.Queue()
        self._recv_thread: threading.Thread | None = None
        self._closed = threading.Event()
        self._send_lock = threading.Lock()

    def connect(self, params: ConnectParams) -> None:
        if params.tcp is None:
            raise ValueError("TCPBackend.connect() requires params.tcp to be set")
        config = params.tcp
        deadline = time.monotonic() + params.connect_timeout
        if config.listen:
            self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._listener.bind((config.host, config.port))
            self._listener.listen(1)
            self._listener.settimeout(params.connect_timeout)
            conn, _addr = self._listener.accept()
            self._sock = conn
        else:
            last_exc: OSError | None = None
            while time.monotonic() < deadline:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.connect((config.host, config.port))
                    self._sock = sock
                    break
                except OSError as exc:
                    last_exc = exc
                    time.sleep(0.2)
            else:
                raise ConnectionError(
                    f"could not connect to {config.host}:{config.port} within "
                    f"{params.connect_timeout}s"
                ) from last_exc

        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

    def _recv_loop(self) -> None:
        sock = self._sock
        assert sock is not None
        try:
            while not self._closed.is_set():
                header = self._recv_exact(sock, _LEN_HEADER.size)
                if header is None:
                    self._recv_queue.put(_CLOSED)
                    return  # peer closed
                (length,) = _LEN_HEADER.unpack(header)
                payload = self._recv_exact(sock, length)
                if payload is None:
                    self._recv_queue.put(_CLOSED)
                    return
                self._recv_queue.put(payload)
        except OSError:
            self._recv_queue.put(_CLOSED)
            return  # socket closed out from under us

    @staticmethod
    def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
        chunks = []
        remaining = n
        while remaining > 0:
            chunk = sock.recv(remaining)
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def send(self, data: bytes) -> None:
        if self._sock is None:
            raise RuntimeError("send() called before connect()")
        framed = _LEN_HEADER.pack(len(data)) + data
        try:
            with self._send_lock:
                self._sock.sendall(framed)
        except OSError as exc:
            raise ConnectionClosedError("TCP send failed: connection is closed") from exc

    def recv(self, timeout: float | None = None) -> bytes:
        try:
            item = self._recv_queue.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"recv() timed out after {timeout}s") from None
        if item is _CLOSED:
            # Put it back so a second caller blocked on recv() also sees it,
            # rather than one caller "consuming" the close signal.
            self._recv_queue.put(_CLOSED)
            raise ConnectionClosedError("TCP connection is closed")
        return item  # type: ignore[return-value]

    def close(self) -> None:
        self._closed.set()
        # shutdown() before close(): if this backend's own _recv_loop thread
        # is currently blocked in sock.recv() on this same fd, close() alone
        # does not reliably unblock it or even guarantee the FIN reaches the
        # peer on Linux (a close() from another thread while a blocking recv()
        # is outstanding on the same fd is a known race - the kernel can leave
        # the connection looking "not really closed" until that recv()
        # returns some other way). shutdown(SHUT_RDWR) acts on the connection
        # itself, unblocks any concurrent recv() on this fd, and reliably
        # sends FIN to the peer regardless of what else references the fd.
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass  # not connected, or peer already gone - fine either way
        for s in (self._sock, self._listener):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass
        if self._recv_thread is not None:
            self._recv_thread.join(timeout=2.0)
