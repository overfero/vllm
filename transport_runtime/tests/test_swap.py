"""Swap test: Phase 2A exit criterion from
`vllm/README_ARCHITECTURE_DECISION.md` ("replace the backend with zero
changes to any call site above ConnectionManager; config values may
change, call sites may not").

`_run_echo` is the one piece of call-site code exercised against every
backend below - only `backend_name` and the backend-specific half of
`ConnectParams` differ between `test_swap_tcp` and `test_swap_udp`.
"""
from __future__ import annotations

import socket
import threading

import pytest

from transport_runtime import (
    ConnectionManager,
    ConnectParams,
    JSONCodec,
    TCPBackendConfig,
    UDPBackendConfig,
)

_ECHO_PAYLOAD = {"hello": "from A", "n": 42}


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_echo(backend_name: str, params_a: ConnectParams, params_b: ConnectParams) -> None:
    """Peer A sends one JSON message, peer B echoes it back, A verifies.
    Both peers run as threads in this process, purely for test convenience
    - nothing here is single-process-specific about the runtime itself."""
    manager_a = ConnectionManager()
    manager_b = ConnectionManager()
    results: dict[str, object] = {}
    errors: list[Exception] = []

    def _peer_a() -> None:
        try:
            conn = manager_a.connect("B", params_a, JSONCodec(), backend_name=backend_name)
            conn.send(_ECHO_PAYLOAD)
            results["echoed"] = conn.recv(timeout=10)
            conn.close()
        except Exception as exc:  # noqa: BLE001 - surfaced to the main thread below
            errors.append(exc)

    def _peer_b() -> None:
        try:
            conn = manager_b.connect("A", params_b, JSONCodec(), backend_name=backend_name)
            msg = conn.recv(timeout=10)
            conn.send(msg)  # echo
            conn.close()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    t_b = threading.Thread(target=_peer_b, daemon=True)
    t_a = threading.Thread(target=_peer_a, daemon=True)
    t_b.start()
    t_a.start()
    t_a.join(timeout=15)
    t_b.join(timeout=15)

    if errors:
        raise errors[0]
    assert results.get("echoed") == _ECHO_PAYLOAD


def test_swap_tcp() -> None:
    port = _free_tcp_port()
    params_a = ConnectParams(
        self_id="A", peer_id="B", tcp=TCPBackendConfig(host="127.0.0.1", port=port, listen=False)
    )
    params_b = ConnectParams(
        self_id="B", peer_id="A", tcp=TCPBackendConfig(host="127.0.0.1", port=port, listen=True)
    )
    _run_echo("tcp", params_a, params_b)


def _signaling_server_reachable(host: str, port: int, timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


_SIGNALING_HOST, _SIGNALING_PORT = "127.0.0.1", 8000


@pytest.mark.skipif(
    not _signaling_server_reachable(_SIGNALING_HOST, _SIGNALING_PORT),
    reason=(
        "UDP swap-test variant needs a live signaling server plus the "
        "external udp_holepunch library (see backends/udp.py) - not assumed "
        "present in every environment this suite runs in. test_swap_tcp is "
        "what proves the swap-test contract when that infra isn't available; "
        "this one proves it end-to-end against the real UDP path when it is."
    ),
)
def test_swap_udp() -> None:
    params_a = ConnectParams(
        self_id="swaptest-A",
        peer_id="swaptest-B",
        udp=UDPBackendConfig(signaling_url=f"http://{_SIGNALING_HOST}:{_SIGNALING_PORT}", port=0),
    )
    params_b = ConnectParams(
        self_id="swaptest-B",
        peer_id="swaptest-A",
        udp=UDPBackendConfig(signaling_url=f"http://{_SIGNALING_HOST}:{_SIGNALING_PORT}", port=0),
    )
    _run_echo("udp", params_a, params_b)
