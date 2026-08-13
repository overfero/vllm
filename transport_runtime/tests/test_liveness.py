"""One failure-injection test, per architecture review's non-blocking
improvement #6: verify `ConnectionClosedError` actually fires rather than
a `recv()` hanging forever or a caller mistaking "closed" for "just slow".
Deliberately TCP-only - TCP can detect a real peer-side close; UDP's
`ConnectionClosedError` guarantee is local-close-only (see
`backends/udp.py`'s module docstring), so a UDP version of this same test
would be testing a guarantee the backend never made.
"""
from __future__ import annotations

import socket
import threading

import pytest

from transport_runtime import ConnectionClosedError, ConnectionManager, ConnectParams, JSONCodec, TCPBackendConfig


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_recv_raises_connection_closed_after_peer_closes() -> None:
    port = _free_tcp_port()
    manager_a = ConnectionManager()
    manager_b = ConnectionManager()
    errors: list[Exception] = []

    def _peer_b_connect_then_close_immediately() -> None:
        try:
            params = ConnectParams(
                self_id="B", peer_id="A", tcp=TCPBackendConfig(host="127.0.0.1", port=port, listen=True)
            )
            conn = manager_b.connect("A", params, JSONCodec(), backend_name="tcp")
            conn.close()  # close right after handshake, before A ever sends anything
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    t_b = threading.Thread(target=_peer_b_connect_then_close_immediately, daemon=True)
    t_b.start()

    params_a = ConnectParams(
        self_id="A", peer_id="B", tcp=TCPBackendConfig(host="127.0.0.1", port=port, listen=False)
    )
    conn_a = manager_a.connect("B", params_a, JSONCodec(), backend_name="tcp")
    t_b.join(timeout=5)
    if errors:
        raise errors[0]

    with pytest.raises(ConnectionClosedError):
        conn_a.recv(timeout=5)
    conn_a.close()
