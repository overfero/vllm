"""Test 2: exchange binary payloads of 1KB/64KB/1MB/16MB and verify
byte-perfect reconstruction, latency, and throughput. Same worker code for
both transports.

Run:
    python3 test2_payload_sizes.py --transport tcp
    python3 test2_payload_sizes.py --transport udp
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time

from _common import MP_CTX, SignalingServer, free_port, transport_config_pair

from vllm.transport import TransportConfig, get_transport

KB = 1024
MB = 1024 * 1024
PAYLOAD_SIZES = [1 * KB, 64 * KB, 1 * MB, 16 * MB]


def _sender(result_queue, backend: str, config: TransportConfig, sizes: list[int]) -> None:
    transport = get_transport(backend)
    transport.connect(config)
    per_size = []
    for size in sizes:
        payload = os.urandom(size)
        digest = hashlib.sha256(payload).hexdigest()
        t0 = time.monotonic()
        transport.send(payload)
        ack = transport.recv()  # receiver echoes back its own digest to confirm byte-perfect receipt
        elapsed = time.monotonic() - t0
        per_size.append({
            "size": size,
            "latency_s": elapsed,
            "throughput_mbps": (size * 8) / (elapsed * 1_000_000) if elapsed > 0 else 0.0,
            "byte_perfect": ack.decode() == digest,
        })
    transport.close()
    result_queue.put({"self_id": config.self_id, "role": "sender", "results": per_size})


def _receiver(result_queue, backend: str, config: TransportConfig, n_messages: int) -> None:
    transport = get_transport(backend)
    transport.connect(config)
    received_sizes = []
    for _ in range(n_messages):
        payload = transport.recv()
        digest = hashlib.sha256(payload).hexdigest()
        received_sizes.append(len(payload))
        transport.send(digest.encode())
    transport.close()
    result_queue.put({"self_id": config.self_id, "role": "receiver", "received_sizes": received_sizes})


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
        cfg_sender, cfg_receiver = transport_config_pair(args.transport, "A", "B", signaling_url, base_port)

        result_queue = MP_CTX.Queue()
        p_recv = MP_CTX.Process(target=_receiver, args=(result_queue, args.transport, cfg_receiver, len(PAYLOAD_SIZES)))
        p_send = MP_CTX.Process(target=_sender, args=(result_queue, args.transport, cfg_sender, PAYLOAD_SIZES))
        p_recv.start()
        p_send.start()

        results = [result_queue.get(timeout=180), result_queue.get(timeout=180)]
        p_send.join(timeout=15)
        p_recv.join(timeout=15)
    finally:
        if signaling is not None:
            signaling.stop()

    sender_result = next(r for r in results if r["role"] == "sender")

    print(f"=== Test 2: payload sizes ({args.transport}) ===")
    ok = True
    for r in sender_result["results"]:
        size_label = f"{r['size'] // KB} KB" if r["size"] < MB else f"{r['size'] // MB} MB"
        print(
            f"  {size_label:>7}: byte-perfect={r['byte_perfect']}  "
            f"latency={r['latency_s'] * 1000:8.2f} ms  throughput={r['throughput_mbps']:8.2f} Mbps"
        )
        ok = ok and r["byte_perfect"]

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
