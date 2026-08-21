"""Tensor phase / Test 3: continuous streaming of 1000 tensors back to back.
Measures throughput, message loss (recv timeout), and corruption (arrived
but didn't match what was sent) for both transports.

Each of the 1000 tensors is seeded deterministically (seed = i) so the
receiver can independently reconstruct what tensor i is supposed to be and
compare, without needing an echo round-trip per tensor (which would turn a
streaming test into 1000 ping-pongs).

Run:
    python3 test8_tensor_streaming.py --transport tcp
    python3 test8_tensor_streaming.py --transport udp
"""
from __future__ import annotations

import argparse
import sys
import time

import torch
from _common import MP_CTX, SignalingServer, free_port, transport_config_pair

from vllm.transport import TransportConfig, get_transport
from vllm.transport.tensor import TransportProcessGroup

N_TENSORS = 1000
N_ELEMENTS = 4096  # 16 KB per tensor (float32) -> ~16.4 MB total


def _make_tensor(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(N_ELEMENTS, generator=g, dtype=torch.float32)


def _sender(result_queue, backend: str, config: TransportConfig) -> None:
    transport = get_transport(backend)
    transport.connect(config)
    pg = TransportProcessGroup(transport)
    t0 = time.monotonic()
    for i in range(N_TENSORS):
        pg.send_tensor(_make_tensor(i))
    elapsed = time.monotonic() - t0
    transport.close()
    total_bytes = N_TENSORS * N_ELEMENTS * 4
    result_queue.put({
        "self_id": config.self_id,
        "role": "sender",
        "elapsed_s": elapsed,
        "throughput_mbps": (total_bytes * 8) / (elapsed * 1_000_000) if elapsed > 0 else 0.0,
    })


def _receiver(result_queue, backend: str, config: TransportConfig) -> None:
    transport = get_transport(backend)
    transport.connect(config)
    pg = TransportProcessGroup(transport)
    lost = 0
    corrupted = 0
    received = 0
    for i in range(N_TENSORS):
        expected = _make_tensor(i)
        try:
            tensor, _stats = pg.recv_tensor(timeout=10)
        except TimeoutError:
            lost += 1
            continue
        received += 1
        if tensor.shape != expected.shape or not torch.equal(tensor, expected):
            corrupted += 1
    transport.close()
    result_queue.put({
        "self_id": config.self_id,
        "role": "receiver",
        "received": received,
        "lost": lost,
        "corrupted": corrupted,
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["tcp", "udp", "quic", "quic-rs"], required=True)
    args = parser.parse_args()

    signaling = SignalingServer() if args.transport in ("udp", "quic", "quic-rs") else None
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

        results = [result_queue.get(timeout=120), result_queue.get(timeout=120)]
        p_send.join(timeout=15)
        p_recv.join(timeout=15)
    finally:
        if signaling is not None:
            signaling.stop()

    sender = next(r for r in results if r["role"] == "sender")
    receiver = next(r for r in results if r["role"] == "receiver")

    print(f"=== Tensor Test 3: streaming {N_TENSORS} tensors ({args.transport}) ===")
    print(f"  elapsed:     {sender['elapsed_s']:.2f} s")
    print(f"  throughput:  {sender['throughput_mbps']:.2f} Mbps")
    print(f"  received:    {receiver['received']}/{N_TENSORS}")
    print(f"  lost:        {receiver['lost']}")
    print(f"  corrupted:   {receiver['corrupted']}")

    ok = receiver["received"] == N_TENSORS and receiver["lost"] == 0 and receiver["corrupted"] == 0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
