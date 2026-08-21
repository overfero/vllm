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
        # Rust-native data path (vllm._rust_udp_raw_engine's send_message/
        # recv_message - sendmmsg/recvmmsg batching, real Go-Back-N
        # retransmission, msg_id-disambiguated streaming) over the same
        # peer.py hole-punch/STUN/keepalive machinery the old aioquic-era
        # UDPTransport used - see udp_rs_transport.py's module docstring
        # and the project_raw_udp_rs_production_readiness memory entry
        # for the three real bugs found and fixed getting here (ack-resend
        # deadlock under loss, ping-pong latency, cross-message
        # corruption). Replaced UDPTransport outright (not added
        # alongside as "udp-rs") per explicit instruction - the old
        # asyncio-per-packet UDPTransport implementation still exists in
        # udp_transport.py if ever needed for comparison, just no longer
        # reachable through this factory.
        from vllm.transport.udp_rs_transport import RawUdpRsTransport

        return RawUdpRsTransport()
    if name == "quic":
        # Rust-native end to end: quinn-proto's protocol state machine AND
        # the whole I/O loop around it (handshake, timers, GSO send,
        # stream framing, drain-before-close) run on a dedicated Rust
        # thread (vllm._rust_quic_engine's PyQuicConnectionDriver) - see
        # quic_transport.py's module docstring and the
        # project_quic_rs_rust_native_driver memory entry for the real
        # deadlock bug found and fixed getting here, and the throughput
        # numbers vs the two backends this replaced. Replaced BOTH the
        # original aioquic-based QUICTransport and the Python-asyncio-
        # orchestrated "quic-rs" (RustQuicTransport) outright per explicit
        # instruction, not added alongside as a third opt-in name -
        # "quic-shared" below went through the identical consolidation
        # (its own old aioquic + Python-orchestrated-Rust variants merged
        # into one Rust-native broker), not a separate capability that
        # was left behind.
        from vllm.transport.quic_transport import QUICTransport

        return QUICTransport()
    if name == "quic-shared":
        # Several logical channels (per-TP-rank PP links + the RPC
        # control channel) multiplexed over ONE real QUIC connection to
        # a given peer machine, instead of "quic"'s one-connection-per-
        # channel - see vllm/transport/quic_broker.py's module docstring.
        # The broker itself (a separate `quic_broker_daemon` process,
        # started ahead of time by stage_server.py/launch_pp_stage.py) is
        # now Rust-native end to end (vllm._rust_quic_engine's
        # PyMultiplexedConnectionDriver), replacing both the old aioquic
        # QuicBroker and the old Python-orchestrated RustQuicBroker
        # outright - same consolidation "quic" went through. This
        # transport class itself is 100% backend-agnostic either way (it
        # only speaks the local Unix-socket IPC protocol, never QUIC
        # directly - see quic_multiplexed_transport.py), so it needed no
        # changes at all. Unlike every other backend here, this one needs
        # its `TransportConfig.extra` populated by the caller with
        # `broker_socket_path`/`channel`.
        from vllm.transport.quic_multiplexed_transport import QuicMultiplexedTransport

        return QuicMultiplexedTransport()
    raise ValueError(f"unknown transport backend {name!r}; available: {list(_KNOWN_BACKENDS)}")
