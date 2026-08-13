"""TensorCodec round-trip correctness. Ported test intent from the
original `vllm/transport/tensor.py` proof (torch.equal round-trip was the
whole point of that phase) - kept here as a real regression test now that
it's a formal Codec rather than free functions.
"""
from __future__ import annotations

import pytest
import torch

from transport_runtime import JSONCodec, TensorCodec


@pytest.mark.parametrize(
    "tensor",
    [
        torch.randn(4, 8),
        torch.zeros(1),
        torch.arange(100).reshape(10, 10),
        torch.randn(2, 3, 5, dtype=torch.float64),
        torch.randn(4, 4, dtype=torch.bfloat16),
        torch.tensor([True, False, True]),
        torch.randn(0, 4),  # empty tensor
    ],
)
def test_tensor_codec_round_trip(tensor: torch.Tensor) -> None:
    codec = TensorCodec()
    encoded = codec.encode(tensor)
    decoded = codec.decode(encoded)
    assert decoded.dtype == tensor.dtype
    assert decoded.shape == tensor.shape
    assert torch.equal(decoded, tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device to construct a GPU tensor")
def test_tensor_codec_rejects_gpu_tensor() -> None:
    codec = TensorCodec()
    with pytest.raises(ValueError, match="CPU tensors"):
        codec.encode(torch.randn(4, device="cuda"))


def test_json_codec_round_trip() -> None:
    codec = JSONCodec()
    payload = {"stage": "ready", "rank": 2, "tags": ["gpu", "t4"]}
    assert codec.decode(codec.encode(payload)) == payload
