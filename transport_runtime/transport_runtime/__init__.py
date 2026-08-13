"""transport_runtime: a framework-agnostic distributed communication runtime.

    from transport_runtime import (
        ConnectionManager, ConnectParams, TCPBackendConfig, UDPBackendConfig,
        TensorCodec, JSONCodec,
    )

    manager = ConnectionManager()
    data_conn = manager.connect(
        "peerB",
        ConnectParams(self_id="A", peer_id="B", tcp=TCPBackendConfig(host="1.2.3.4", port=30000)),
        TensorCodec(),
        backend_name="tcp",
        role="data",
    )
    data_conn.send(my_tensor)
    tensor_back = data_conn.recv()

This package is the Phase 2A extraction described in
`vllm/README_ARCHITECTURE_DECISION.md` (in the sibling `vllm/` checkout):
vLLM, SGLang, or any other inference/training framework are customers of
this runtime via a thin adapter - none of this package imports or knows
about any of them. See README.md in this directory for the full Non-Goals
list and layering rationale.
"""
from transport_runtime.backend import (
    Backend,
    BackendError,
    ConnectionClosedError,
    ConnectParams,
    TCPBackendConfig,
    UDPBackendConfig,
)
from transport_runtime.codec import BytesCodec, Codec, JSONCodec, TensorCodec
from transport_runtime.connection import Connection, ConnectionManager
from transport_runtime.factory import get_backend, register_backend

__all__ = [
    "Backend",
    "BackendError",
    "ConnectionClosedError",
    "ConnectParams",
    "TCPBackendConfig",
    "UDPBackendConfig",
    "Codec",
    "TensorCodec",
    "JSONCodec",
    "BytesCodec",
    "Connection",
    "ConnectionManager",
    "get_backend",
    "register_backend",
]
