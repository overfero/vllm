"""Plain-TCP implementation of the Transport interface.

Deliberately independent of vLLM's real torch.distributed/NCCL TCPStore
usage (vllm/utils/network_utils.py, vllm/distributed/parallel_state.py) -
this is a standalone data-plane transport for the abstraction being
introduced, not a wrapper around the existing rendezvous machinery. See
vllm/transport/README.md for why that scope line was drawn here.
"""
from __future__ import annotations

import queue
import socket
import struct
import threading
import time

from vllm.transport.base import Transport, TransportConfig

_LEN_HEADER = struct.Struct("!Q")  # 8-byte big-endian message length prefix


class TCPTransport(Transport):
    def __init__(self) -> None:
        self._sock: socket.socket | None = None
        self._listener: socket.socket | None = None
        self._recv_queue: "queue.Queue[bytes]" = queue.Queue()
        self._recv_thread: threading.Thread | None = None
        self._closed = threading.Event()
        self._send_lock = threading.Lock()

    def connect(self, config: TransportConfig) -> None:
        deadline = time.monotonic() + config.connect_timeout
        if config.listen:
            self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._listener.bind((config.host, config.port))
            self._listener.listen(1)
            self._listener.settimeout(config.connect_timeout)
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
                    f"{config.connect_timeout}s"
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
                    return  # peer closed
                (length,) = _LEN_HEADER.unpack(header)
                payload = self._recv_exact(sock, length)
                if payload is None:
                    return
                self._recv_queue.put(payload)
        except OSError:
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
        with self._send_lock:
            self._sock.sendall(framed)

    def recv(self, timeout: float | None = None) -> bytes:
        try:
            return self._recv_queue.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"recv() timed out after {timeout}s") from None

    def close(self) -> None:
        self._closed.set()
        for s in (self._sock, self._listener):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass
        if self._recv_thread is not None:
            self._recv_thread.join(timeout=2.0)
