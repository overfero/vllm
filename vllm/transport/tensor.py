"""Tensor exchange on top of the Transport abstraction.

Transport only knows about `bytes` (see `base.py`). This module is the layer
above it that knows about `torch.Tensor`: it turns a tensor into
`metadata + raw bytes`, hands that to `Transport.send()`/`recv()` as one
message, and reconstructs a tensor on the other side.

Deliberately NOT touching `UDPTransport`/`TCPTransport`/`GroupCoordinator`/
`StatelessProcessGroup` - see `README.md` ("Tensor communication phase") for
why a new `TransportProcessGroup` was introduced here instead of extending
one of those.

CPU tensors only. No zero-copy, no GPUDirect, no optimization - correctness
(`torch.equal`/`torch.allclose` round-trip) is the only goal of this phase.
"""
from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass

import torch

from vllm.transport.base import Transport

_HEADER_LEN = struct.Struct("!I")  # length of the JSON metadata block that follows


def serialize_tensor(tensor: torch.Tensor) -> bytes:
    """Tensor -> (metadata, raw bytes) -> one flat bytes message.

    Metadata is dtype + shape, JSON-encoded and length-prefixed. The tensor
    payload is reinterpreted as raw uint8 bytes via `.view(torch.uint8)`,
    which works for any dtype (including bfloat16, which numpy can't
    represent) without a dtype-specific conversion table.
    """
    if tensor.device.type != "cpu":
        raise ValueError(f"serialize_tensor only supports CPU tensors, got device={tensor.device}")

    tensor = tensor.contiguous()
    dtype_name = str(tensor.dtype).removeprefix("torch.")
    meta = json.dumps({"dtype": dtype_name, "shape": list(tensor.shape)}).encode()

    raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    return _HEADER_LEN.pack(len(meta)) + meta + raw


def deserialize_tensor(data: bytes) -> torch.Tensor:
    """Inverse of `serialize_tensor`. Reconstructs dtype and shape exactly."""
    (meta_len,) = _HEADER_LEN.unpack_from(data, 0)
    offset = _HEADER_LEN.size
    meta = json.loads(data[offset:offset + meta_len])
    offset += meta_len

    dtype = getattr(torch, meta["dtype"])
    shape = tuple(meta["shape"])
    raw = bytearray(data[offset:])  # copy: torch.frombuffer requires a writable buffer

    flat_bytes = torch.frombuffer(raw, dtype=torch.uint8)
    return flat_bytes.view(dtype).reshape(shape).clone()


@dataclass
class TensorSendStats:
    serialize_s: float
    transport_s: float
    nbytes: int


@dataclass
class TensorRecvStats:
    transport_s: float
    deserialize_s: float
    nbytes: int


class TransportProcessGroup:
    """Minimal 2-rank point-to-point tensor exchange, backed by a `Transport`.

    Analogous in spirit to `StatelessProcessGroup` (a lightweight,
    non-torch.distributed way to move data between two ranks) but its data
    plane is the `Transport` abstraction instead of a `TCPStore`. It does not
    call `torch.distributed.init_process_group`, does not construct a
    `ProcessGroup`, and does not know about NCCL/gloo/CUDA - it just
    serializes/deserializes tensors around `transport.send()`/`recv()`.

    Deliberately single-peer (send/recv, no ranks/world_size/collectives):
    that is all this phase needs to prove, and it's what `Transport` itself
    supports. See README.md deliverable 4 for what a full N-rank
    `ProcessGroup` replacement would additionally require.
    """

    def __init__(self, transport: Transport) -> None:
        self.transport = transport

    def send_tensor(self, tensor: torch.Tensor) -> TensorSendStats:
        t0 = time.monotonic()
        payload = serialize_tensor(tensor)
        t1 = time.monotonic()
        self.transport.send(payload)
        t2 = time.monotonic()
        return TensorSendStats(serialize_s=t1 - t0, transport_s=t2 - t1, nbytes=len(payload))

    def recv_tensor(self, timeout: float | None = None) -> tuple[torch.Tensor, TensorRecvStats]:
        t0 = time.monotonic()
        payload = self.transport.recv(timeout=timeout)
        t1 = time.monotonic()
        tensor = deserialize_tensor(payload)
        t2 = time.monotonic()
        return tensor, TensorRecvStats(transport_s=t1 - t0, deserialize_s=t2 - t1, nbytes=len(payload))
