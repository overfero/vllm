"""Tensor phase / Test 4: bidirectional tensor streaming. Worker A sends a
tensor and receives a tensor; Worker B receives a tensor and sends a
tensor - both directions running at the same time on each side (separate
send/recv threads over the same connection), not a sequential ping-pong.

Run:
    python3 test9_tensor_bidirectional.py --transport tcp
    python3 test9_tensor_bidirectional.py --transport udp
"""
from __future__ import annotations

import argparse
import sys
import threading
import time

import torch
from _common import MP_CTX, SignalingServer, free_port, transport_config_pair

from vllm.transport import TransportConfig, get_transport
from vllm.transport.tensor import TransportProcessGroup

N_ELEMENTS = 1 * 1024 * 1024 // 4  # 1 MB, float32


def _make_tensor(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(N_ELEMENTS, generator=g, dtype=torch.float32)


def _worker(result_queue, backend: str, config: TransportConfig, my_seed: int, peer_seed: int) -> None:
    transport = get_transport(backend)
    transport.connect(config)
    pg = TransportProcessGroup(transport)

    my_tensor = _make_tensor(my_seed)
    expected_from_peer = _make_tensor(peer_seed)
    outcome: dict = {}

    def _do_send() -> None:
        outcome["send_stats"] = pg.send_tensor(my_tensor)

    def _do_recv() -> None:
        tensor, stats = pg.recv_tensor(timeout=30)
        outcome["received"] = tensor
        outcome["recv_stats"] = stats

    t_send = threading.Thread(target=_do_send)
    t_recv = threading.Thread(target=_do_recv)
    t0 = time.monotonic()
    t_send.start()
    t_recv.start()
    t_send.join()
    t_recv.join()
    elapsed = time.monotonic() - t0
    transport.close()

    received = outcome["received"]
    result_queue.put({
        "self_id": config.self_id,
        "elapsed_s": elapsed,
        "equal": torch.equal(received, expected_from_peer),
        "shape_ok": tuple(received.shape) == tuple(expected_from_peer.shape),
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["tcp", "udp"], required=True)
    args = parser.parse_args()

    signaling = SignalingServer() if args.transport == "udp" else None
    signaling_url = None
    if signaling is not None:
        signaling.start()
        signaling_url = signaling.url

    try:
        base_port = free_port()
        cfg_a, cfg_b = transport_config_pair(args.transport, "A", "B", signaling_url, base_port)
        seed_a, seed_b = 111, 222

        result_queue = MP_CTX.Queue()
        p_a = MP_CTX.Process(target=_worker, args=(result_queue, args.transport, cfg_a, seed_a, seed_b))
        p_b = MP_CTX.Process(target=_worker, args=(result_queue, args.transport, cfg_b, seed_b, seed_a))
        p_a.start()
        p_b.start()

        results = [result_queue.get(timeout=60), result_queue.get(timeout=60)]
        p_a.join(timeout=15)
        p_b.join(timeout=15)
    finally:
        if signaling is not None:
            signaling.stop()

    print(f"=== Tensor Test 4: bidirectional simultaneous exchange ({args.transport}) ===")
    ok = True
    for r in sorted(results, key=lambda x: x["self_id"]):
        print(f"  {r['self_id']}: elapsed={r['elapsed_s']:.3f}s  shape_ok={r['shape_ok']}  equal={r['equal']}")
        ok = ok and r["shape_ok"] and r["equal"]

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
