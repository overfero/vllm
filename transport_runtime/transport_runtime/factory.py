"""Single place backend selection happens.

Nowhere else should branch on `if backend == "tcp"` / `if backend == "udp"`
- ask `get_backend()` for a ready-to-use `Backend` instance instead. This is
what makes the swap test meaningful: every call site above this module only
ever sees a `Backend`, never a backend name, so adding a new backend (QUIC,
say) never touches those call sites - only `register_backend()` needs to be
called once, wherever that new backend is defined.
"""
from __future__ import annotations

from collections.abc import Callable

from transport_runtime.backend import Backend

_REGISTRY: dict[str, Callable[[], Backend]] = {}


def register_backend(name: str, factory: Callable[[], Backend]) -> None:
    """Register a backend constructor under `name`.

    Built-in backends (`tcp`, `udp`) register themselves via this same
    function on first use of `get_backend()` — see `_ensure_builtins()`.
    Third-party/future backends (QUIC, RDMA, shared memory, ...) use this
    directly and never need to modify this file.
    """
    _REGISTRY[name.lower()] = factory


def _ensure_builtins() -> None:
    if "tcp" not in _REGISTRY:
        from transport_runtime.backends.tcp import TCPBackend

        register_backend("tcp", TCPBackend)
    if "udp" not in _REGISTRY:
        from transport_runtime.backends.udp import UDPBackend

        register_backend("udp", UDPBackend)


def get_backend(name: str) -> Backend:
    """Return a fresh, unconnected `Backend` instance for `name`.

    Raises `ValueError` for an unknown name, listing what is currently
    registered.
    """
    _ensure_builtins()
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"unknown transport backend {name!r}; available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key]()
