"""Backend abstraction: the lowest layer of the transport runtime.

A `Backend` moves opaque byte-string messages between exactly two named
endpoints. It knows nothing about tensors, topology, pipelines, ranks,
training vs. inference, or scheduling — see README.md's Non-Goals. Nothing
above this layer (Codec, Connection, ConnectionManager, any framework
adapter) is allowed to reach into a `Backend` for anything but
`connect`/`send`/`recv`/`close`.

This is the extraction of what shipped, proven, as vLLM's own
`vllm/transport/base.py` + `tcp_transport.py` + `udp_transport.py` (run
successfully across 3 real NAT'd machines). Two things changed on the way
out, both closing gaps found in architecture review before this extraction:

1. `TransportConfig` (one flat dataclass mixing TCP fields and UDP fields,
   each backend silently ignoring the other's fields) is replaced by
   `ConnectParams`, which carries backend-specific config in nested
   `TCPBackendConfig`/`UDPBackendConfig` objects. A `TCPBackend` now
   physically cannot see UDP fields and vice versa — the old design's
   "swap backend, call sites don't change" claim was true of call sites
   but not of the config object's own shape; this makes it true of the
   shape too.
2. The error/liveness contract, previously implicit (recv() timing out
   looked identical whether "nothing has arrived yet" or "the peer is
   gone and nothing ever will"), is now explicit: see `recv()`'s
   docstring and `ConnectionClosedError` below.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class BackendError(Exception):
    """Base class for all transport_runtime backend errors."""


class ConnectionClosedError(BackendError):
    """Raised by `recv()`/`send()` when the connection is known to be
    closed — locally (`close()` was called) or, where a backend can
    detect it, by the peer. Distinct from `TimeoutError`, which means
    only "nothing arrived within the wait", not "nothing ever will."

    Honesty note: UDP is best-effort and has no reliable way to detect a
    silent peer-side disconnect (there is no FIN). `UDPBackend` only
    raises this for a *local* `close()`; a UDP peer that vanishes without
    telling anyone will still surface as a plain `TimeoutError` on the
    next `recv()`, indefinitely. `TCPBackend` can and does detect the
    peer-side case too, because TCP actually has a close signal.
    """


@dataclass
class TCPBackendConfig:
    host: str = "127.0.0.1"
    port: int = 0
    listen: bool = False  # one side must listen, the other connects


@dataclass
class UDPBackendConfig:
    signaling_url: str = "http://127.0.0.1:8000"
    # "stun" (not "preserve") is the correct default for any real NAT that
    # doesn't preserve the local bind port on outbound mapping - which
    # includes at least GCP Cloud NAT, confirmed via a live 3-real-machine
    # test (see vllm/transport/pipeline_bootstrap.py's UDPBackendConfig
    # call site for the full incident writeup). "preserve" is kept
    # available, not removed, for callers with a real port-preserving NAT
    # (e.g. manual 1:1 port forwarding) who want to skip the STUN round
    # trip - but it is no longer the silent default.
    mode: str = "stun"  # "stun" or "preserve" - see backends/udp.py
    port: int = 0
    stun_host: str = "stun.l.google.com"
    stun_port: int = 19302


@dataclass
class ConnectParams:
    """Everything needed to connect to one named peer.

    `tcp`/`udp` are populated only for the backend actually in use — a
    `TCPBackend.connect()` reads `params.tcp` and never touches
    `params.udp` (and raises `ValueError` if `params.tcp` is `None`,
    rather than silently defaulting), so there is no way for TCP-specific
    and UDP-specific configuration to leak into each other's code paths.
    """

    self_id: str
    peer_id: str
    connect_timeout: float = 60.0
    tcp: TCPBackendConfig | None = None
    udp: UDPBackendConfig | None = None
    extra: dict = field(default_factory=dict)


class Backend(ABC):
    """Minimal point-to-point transport contract.

    Every implementation supports exactly this: connect to one named
    peer, exchange whole byte-string messages, and close cleanly. Nothing
    here knows about tensors, ranks, process groups, model execution, or
    the shape of the network beyond this one peer — that is what makes it
    safe to build a `Codec`/`Connection`/`ConnectionManager` on top
    without the boundary leaking back down.
    """

    @abstractmethod
    def connect(self, params: ConnectParams) -> None:
        """Establish a connection to the peer described by `params`.

        Blocks until the connection is ready to send/recv. Raises
        `ConnectionError` or `TimeoutError` on failure — there is no
        partial-connected state a caller needs to check for.
        """

    @abstractmethod
    def send(self, data: bytes) -> None:
        """Send one complete message. Blocks until handed off to the
        backend.

        Raises `RuntimeError` if called before `connect()` returns, and
        `ConnectionClosedError` if the backend already knows the
        connection is gone.
        """

    @abstractmethod
    def recv(self, timeout: float | None = None) -> bytes:
        """Block until one complete message has arrived, and return it.

        Raises `TimeoutError` if `timeout` (seconds) elapses with nothing
        arriving. Raises `ConnectionClosedError` instead if the
        connection is known-closed (see that class's docstring for what
        "known" means per backend) — callers that want to distinguish
        "keep waiting" from "stop, this peer is gone" should catch these
        separately.
        """

    @abstractmethod
    def close(self) -> None:
        """Release all resources (sockets, background threads/loops).

        Idempotent — safe to call more than once, never raises.
        """

    def __enter__(self) -> "Backend":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
