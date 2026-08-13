"""Connection and ConnectionManager: the layer that owns liveness/reconnect
for one point-to-point link, and a flat registry of such links.

Deliberately absent from this module, on purpose (see README.md's
Non-Goals and the architecture-review history that removed it from an
earlier draft): any notion of topology, edges, "this is a pipeline", or
node capability. `ConnectionManager` only ever knows `peer_id -> Connection`
- who is connected to whom overall, and in what shape (chain, mesh,
whatever), is knowledge that belongs to whichever adapter called
`connect()` for each peer, never to the runtime. Keeping the runtime
blind to that shape is what avoids a dual-source-of-truth bug class this
project has already hit once, for an unrelated reason, in the original
vLLM fork (two representations of the same tensor name needing manual
reconciliation - see README_LIVE_DEPLOYMENT_LOG.md Bugs 2-3 in the vllm
checkout this package was extracted from).
"""
from __future__ import annotations

import threading
from typing import Literal

from transport_runtime.backend import Backend, ConnectParams
from transport_runtime.codec import Codec
from transport_runtime.factory import get_backend

Role = Literal["control", "data"]


class Connection:
    """One point-to-point link: a `Backend` plus a `Codec`, nothing else.

    A `Connection` does not know about any peer other than the one it is
    connected to - no routing, no `next`/`prev`, no awareness that it
    might be one of several links a `ConnectionManager` is holding.
    """

    def __init__(self, backend: Backend, codec: Codec, *, peer_id: str, role: Role) -> None:
        self.backend = backend
        self.codec = codec
        self.peer_id = peer_id
        self.role: Role = role

    def send(self, obj: object) -> None:
        self.backend.send(self.codec.encode(obj))

    def recv(self, timeout: float | None = None) -> object:
        return self.codec.decode(self.backend.recv(timeout=timeout))

    def close(self) -> None:
        self.backend.close()

    def __enter__(self) -> "Connection":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class ConnectionManager:
    """Flat `(peer_id, role) -> Connection` registry.

    Two `Connection`s per peer is the common case (see README.md Part 5's
    converged layering: one control-plane link, reliable/ordered/small,
    typically a TCP backend; one data-plane link, throughput-oriented,
    typically UDP) but nothing here requires exactly two - `role` is just
    a label distinguishing multiple links to the *same* peer_id, not a
    graph edge to a *different* peer. Establishing links to other peers
    (and knowing that a set of peer_ids together form some larger
    structure) is the caller's job, one `connect()` call at a time.
    """

    def __init__(self) -> None:
        self._connections: dict[tuple[str, Role], Connection] = {}
        self._lock = threading.Lock()

    def connect(
        self,
        peer_id: str,
        params: ConnectParams,
        codec: Codec,
        *,
        backend_name: str,
        role: Role = "data",
    ) -> Connection:
        """Establish (or replace) the `role` connection to `peer_id`.

        Blocks until connected or raises - same contract as
        `Backend.connect()`, since that is exactly what this calls.
        """
        backend = get_backend(backend_name)
        backend.connect(params)
        conn = Connection(backend, codec, peer_id=peer_id, role=role)
        with self._lock:
            self._connections[(peer_id, role)] = conn
        return conn

    def get(self, peer_id: str, role: Role = "data") -> Connection | None:
        with self._lock:
            return self._connections.get((peer_id, role))

    def peer_ids(self) -> list[str]:
        """All distinct peer_ids currently registered, in no particular
        order and with no claim about how they relate to each other -
        see class docstring."""
        with self._lock:
            return sorted({peer_id for peer_id, _role in self._connections})

    def close(self, peer_id: str, role: Role = "data") -> None:
        with self._lock:
            conn = self._connections.pop((peer_id, role), None)
        if conn is not None:
            conn.close()

    def close_all(self) -> None:
        with self._lock:
            conns = list(self._connections.values())
            self._connections.clear()
        for conn in conns:
            conn.close()
