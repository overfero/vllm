"""Single place transport backend selection happens.

Nowhere else in vLLM (or in code using this package) should branch on
`if transport == "tcp"` / `if transport == "udp"` - ask this function for a
ready-to-use Transport instance instead. Resolution order: explicit
`backend` argument, then the VLLM_TRANSPORT env var (also settable via the
`--transport` CLI flag, see vllm/config/parallel.py), then "tcp".
"""
from __future__ import annotations

import vllm.envs as envs
from vllm.transport.base import Transport

_KNOWN_BACKENDS = ("tcp", "udp")


def get_transport(backend: str | None = None) -> Transport:
    name = (backend or envs.VLLM_TRANSPORT).lower()
    if name == "tcp":
        from vllm.transport.tcp_transport import TCPTransport

        return TCPTransport()
    if name == "udp":
        from vllm.transport.udp_transport import UDPTransport

        return UDPTransport()
    raise ValueError(f"unknown transport backend {name!r}; available: {list(_KNOWN_BACKENDS)}")
