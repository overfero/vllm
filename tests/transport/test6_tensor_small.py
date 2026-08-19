"""Tensor phase / Test 1: small tensors (16/64/256/1024 elements), several
dtypes, random values. Verifies exact round-trip through
TransportProcessGroup.send_tensor()/recv_tensor() for both transports.

No echo round-trip needed for verification: sender and receiver both derive
the "expected" tensor for message i from the same seed, independently, so
the receiver can compare what arrived against what it *knows* was sent
without the sender ever transmitting a reference copy.

Run:
    python3 test6_tensor_small.py --transport tcp
    python3 test6_tensor_small.py --transport udp
"""
from __future__ import annotations

import argparse
import sys

import torch
from _common import MP_CTX, SignalingServer, free_port, transport_config_pair

from vllm.transport import TransportConfig, get_transport
from vllm.transport.tensor import TransportProcessGroup

SIZES = [16, 64, 256, 1024]
DTYPES = [torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.int32, torch.bool]


def _make_tensor(seed: int, n: int, dtype: torch.dtype) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    if dtype is torch.bool:
        return torch.randint(0, 2, (n,), generator=g, dtype=torch.int64).bool()
    if dtype in (torch.int32, torch.int64):
        return torch.randint(-1000, 1000, (n,), generator=g, dtype=dtype)
    return torch.randn(n, generator=g).to(dtype)


def _cases() -> list[tuple[int, int, torch.dtype]]:
    return [(i, n, dt) for i, (n, dt) in enumerate((n, dt) for n in SIZES for dt in DTYPES)]


def _sender(result_queue, backend: str, config: TransportConfig) -> None:
    transport = get_transport(backend)
    transport.connect(config)
    pg = TransportProcessGroup(transport)
    for seed, n, dtype in _cases():
        pg.send_tensor(_make_tensor(seed, n, dtype))
    transport.close()
    result_queue.put({"self_id": config.self_id, "role": "sender"})


def _receiver(result_queue, backend: str, config: TransportConfig) -> None:
    transport = get_transport(backend)
    transport.connect(config)
    pg = TransportProcessGroup(transport)
    per_case = []
    for seed, n, dtype in _cases():
        expected = _make_tensor(seed, n, dtype)
        received, _stats = pg.recv_tensor(timeout=30)
        per_case.append({
            "n": n,
            "dtype": str(dtype),
            "shape_ok": tuple(received.shape) == tuple(expected.shape),
            "dtype_ok": received.dtype == expected.dtype,
            "equal": torch.equal(received, expected),
        })
    transport.close()
    result_queue.put({"self_id": config.self_id, "role": "receiver", "results": per_case})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["tcp", "udp", "quic"], required=True)
    args = parser.parse_args()

    signaling = SignalingServer() if args.transport in ("udp", "quic") else None
    signaling_url = None
    if signaling is not None:
        signaling.start()
        signaling_url = signaling.url

    try:
        base_port = free_port()
        cfg_sender, cfg_receiver = transport_config_pair(args.transport, "A", "B", signaling_url, base_port)

        result_queue = MP_CTX.Queue()
        p_recv = MP_CTX.Process(target=_receiver, args=(result_queue, args.transport, cfg_receiver))
        p_send = MP_CTX.Process(target=_sender, args=(result_queue, args.transport, cfg_sender))
        p_recv.start()
        p_send.start()

        results = [result_queue.get(timeout=60), result_queue.get(timeout=60)]
        p_send.join(timeout=15)
        p_recv.join(timeout=15)
    finally:
        if signaling is not None:
            signaling.stop()

    receiver_result = next(r for r in results if r["role"] == "receiver")

    print(f"=== Tensor Test 1: small tensors ({args.transport}) ===")
    ok = True
    for r in receiver_result["results"]:
        status = "OK" if (r["shape_ok"] and r["dtype_ok"] and r["equal"]) else "MISMATCH"
        print(f"  n={r['n']:5d} dtype={r['dtype']:>16}: {status}")
        ok = ok and r["shape_ok"] and r["dtype_ok"] and r["equal"]

    print(f"  {len(receiver_result['results'])} cases checked")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
