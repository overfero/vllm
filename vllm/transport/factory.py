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

_KNOWN_BACKENDS = ("tcp", "udp", "quic", "quic-shared")


def get_transport(backend: str | None = None) -> Transport:
    name = (backend or envs.VLLM_TRANSPORT).lower()
    if name == "tcp":
        from vllm.transport.tcp_transport import TCPTransport

        return TCPTransport()
    if name == "udp":
        from vllm.transport.udp_transport import UDPTransport

        return UDPTransport()
    if name == "quic":
        from vllm.transport.quic_transport import QUICTransport

        return QUICTransport()
    if name == "quic-shared":
        # Several logical channels (per-TP-rank PP links + the RPC
        # control channel) multiplexed over ONE real QUIC connection to
        # a given peer machine, instead of "quic"'s one-connection-per-
        # channel - see vllm/transport/quic_broker.py's module docstring.
        # Unlike every other backend here, this one needs its
        # `TransportConfig.extra` populated by the caller with
        # `broker_socket_path`/`channel` (see quic_multiplexed_transport.py)
        # - the actual QUIC connection is owned by a separate
        # `quic_broker_daemon` process, started ahead of time by
        # stage_server.py/launch_pp_stage.py, not by this call.
        from vllm.transport.quic_multiplexed_transport import QuicMultiplexedTransport

        return QuicMultiplexedTransport()
    raise ValueError(f"unknown transport backend {name!r}; available: {list(_KNOWN_BACKENDS)}")
