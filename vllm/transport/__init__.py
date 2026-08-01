"""Swappable worker-to-worker transport abstraction.

    from vllm.transport import get_transport, TransportConfig

    transport = get_transport()  # picks backend via VLLM_TRANSPORT / --transport
    transport.connect(TransportConfig(self_id="A", peer_id="B", ...))
    transport.send(b"hello")
    transport.recv()
    transport.close()

See README.md in this directory for scope, and vllm/transport/base.py for
the full interface. TCPTransport and UDPTransport are intentionally not
imported here - import them directly from their modules if you need the
concrete class rather than going through get_transport().
"""
from vllm.transport.base import Transport, TransportConfig
from vllm.transport.factory import get_transport

__all__ = ["Transport", "TransportConfig", "get_transport"]
