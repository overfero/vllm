"""Transport abstraction for worker-to-worker communication.

This is intentionally decoupled from vLLM's real distributed runtime
(torch.distributed / NCCL / gloo). It exists to prove that worker-to-worker
communication can be expressed behind a swappable interface, independent of
model execution, CUDA, or tensor/pipeline-parallel scheduling. See
`vllm/transport/README.md` for scope and how this relates to the rest of
vLLM's distributed stack.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class TransportConfig:
    """Everything either transport backend might need to connect to a peer.

    Fields not relevant to a given backend are simply ignored by it (e.g.
    TCPTransport ignores signaling_url/mode; UDPTransport ignores host/listen).
    Keeping one shape for both backends is what lets test code stay identical
    across `--transport tcp` and `--transport udp`.
    """

    self_id: str
    peer_id: str

    # TCPTransport
    host: str = "127.0.0.1"
    port: int = 0
    listen: bool = False  # one side must listen, the other connects

    # UDPTransport (adapts the existing hole-punch transport)
    signaling_url: str = "http://127.0.0.1:8000"
    udp_mode: str = "preserve"  # "stun" or "preserve" - see existing transport's README
    udp_port: int = 0
    stun_host: str = "stun.l.google.com"
    stun_port: int = 19302
    connect_timeout: float = 60.0

    # QUICTransport (also reuses signaling_url/udp_mode/udp_port/stun_*/
    # connect_timeout/listen above - hole punch is shared with UDPTransport,
    # see quic_transport.py's module docstring)
    quic_idle_timeout: float = 45.0  # seconds with no traffic before the QUIC
    # connection itself declares the peer dead (independent of, and a real
    # improvement over, UDPTransport's total lack of dead-peer detection)
    quic_max_message_bytes: int = 2 * 1024 * 1024 * 1024  # application-level
    # cap on top of QUIC's own flow control - defense in depth, not relying
    # solely on library defaults for this (see quic_transport.py)

    extra: dict = field(default_factory=dict)


class Transport(ABC):
    """Minimal point-to-point transport contract.

    Every implementation must support exactly this: connect to one named
    peer, exchange whole byte-string messages, and close cleanly. Nothing
    here knows about tensors, ranks, process groups, or model execution -
    that's deliberate, so this can be validated as pure communication
    infrastructure before anything tries to build tensor-parallel or
    pipeline-parallel semantics on top of it.
    """

    @abstractmethod
    def connect(self, config: TransportConfig) -> None:
        """Establish a connection to the peer described by `config`.

        Blocks until the connection is ready to send/recv, or raises on
        failure/timeout.
        """

    @abstractmethod
    def send(self, data: bytes) -> None:
        """Send one complete message. Blocks until handed off to the transport."""

    @abstractmethod
    def recv(self, timeout: float | None = None) -> bytes:
        """Block until one complete message has arrived, and return it.

        If `timeout` (seconds) is given and no message arrives in time,
        raises `TimeoutError`.
        """

    @abstractmethod
    def close(self) -> None:
        """Release all resources (sockets, background threads/loops)."""

    def __enter__(self) -> "Transport":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
