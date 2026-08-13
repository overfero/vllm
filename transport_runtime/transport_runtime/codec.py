"""Codec: the layer between a `Backend` (which only knows `bytes`) and a
`Connection` (which callers use with real objects - tensors, dicts,
whatever the adapter needs to move).

A `Backend` is never allowed to know a Codec exists; a `Codec` is never
allowed to know a Backend exists. Everything below this line is generic -
see README.md's Non-Goals: the runtime does not interpret payload
semantics, and that boundary lives exactly here.

Known limitation, stated rather than hidden (raised during architecture
review): `Codec.encode()/decode()` is a `bytes`-in/`bytes`-out contract.
For CPU tensors (this package's only proven case so far - see
`TensorCodec`) that means a real memory copy on every call. That's fine
for correctness and for the workloads this has been validated against,
but it is not zero-copy, and pinned-memory/GPU-staging/buffer-protocol
variants are explicitly out of scope for this extraction - flagged for
Phase 6 (performance tuning) if profiling shows it matters, not solved
here speculatively.
"""
from __future__ import annotations

import json
import struct
from typing import Protocol, runtime_checkable

import torch

_HEADER_LEN = struct.Struct("!I")  # length of the JSON metadata block that follows


@runtime_checkable
class Codec(Protocol):
    """Anything that can turn one Python object into bytes and back.

    `TensorCodec` is the default implementation this package ships with,
    because tensor payloads are what pipeline-parallel activation passing
    actually needs today - but nothing about `Connection`/
    `ConnectionManager` assumes tensors specifically; any `Codec` works.
    """

    def encode(self, obj: object) -> bytes:
        """Serialize `obj` to one flat `bytes` message."""
        ...

    def decode(self, data: bytes) -> object:
        """Inverse of `encode()`."""
        ...


class TensorCodec:
    """CPU `torch.Tensor` <-> bytes. Ported from vLLM's
    `vllm/transport/tensor.py` (`serialize_tensor`/`deserialize_tensor`),
    unchanged logic - ownership moved here so the Codec boundary is a
    formal interface point instead of two free functions.
    """

    def encode(self, tensor: torch.Tensor) -> bytes:
        if tensor.device.type != "cpu":
            raise ValueError(
                f"TensorCodec only supports CPU tensors, got device={tensor.device}"
            )
        tensor = tensor.contiguous()
        dtype_name = str(tensor.dtype).removeprefix("torch.")
        meta = json.dumps({"dtype": dtype_name, "shape": list(tensor.shape)}).encode()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return _HEADER_LEN.pack(len(meta)) + meta + raw

    def decode(self, data: bytes) -> torch.Tensor:
        (meta_len,) = _HEADER_LEN.unpack_from(data, 0)
        offset = _HEADER_LEN.size
        meta = json.loads(data[offset : offset + meta_len])
        offset += meta_len

        dtype = getattr(torch, meta["dtype"])
        shape = tuple(meta["shape"])
        raw = bytearray(data[offset:])  # copy: torch.frombuffer requires a writable buffer

        if not raw:
            # torch.frombuffer rejects a zero-length buffer outright (found
            # via this package's own test suite, not present in the
            # original code's test coverage) - an empty tensor (any dim==0)
            # has nothing to reconstruct from bytes, so build it directly.
            return torch.empty(shape, dtype=dtype)

        flat_bytes = torch.frombuffer(raw, dtype=torch.uint8)
        return flat_bytes.view(dtype).reshape(shape).clone()


class JSONCodec:
    """Small structured messages - the natural fit for a control-plane
    Connection (health, barrier, cancel, stage-ready, ...): reliable,
    ordered, small, latency-insensitive, exactly the profile a control
    channel needs, and exactly what tensor payloads don't need. Kept
    intentionally trivial; do not grow this into a schema/versioning
    system unless a concrete adapter actually needs one.
    """

    def encode(self, obj: object) -> bytes:
        return json.dumps(obj).encode("utf-8")

    def decode(self, data: bytes) -> object:
        return json.loads(data.decode("utf-8"))


class BytesCodec:
    """Identity codec, for callers that already have bytes and don't want
    a serialization step at all."""

    def encode(self, obj: object) -> bytes:
        if not isinstance(obj, (bytes, bytearray)):
            raise TypeError(f"BytesCodec.encode() expects bytes, got {type(obj)}")
        return bytes(obj)

    def decode(self, data: bytes) -> bytes:
        return data
