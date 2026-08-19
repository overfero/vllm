"""Test 1: Worker A sends "hello" and receives "world"; Worker B receives
"hello" and sends "world". Communication only - no CUDA, no model.

Run:
    python3 test1_hello_world.py --transport tcp
    python3 test1_hello_world.py --transport udp

_worker_a / _worker_b below are 100% identical for both transports - only
the TransportConfig built in main() differs.
"""
from __future__ import annotations

import argparse
import sys
import time

from _common import MP_CTX, SignalingServer, free_port, transport_config_pair

from vllm.transport import TransportConfig, get_transport


def _worker_a(result_queue, backend: str, config: TransportConfig) -> None:
    transport = get_transport(backend)
    t0 = time.monotonic()
    transport.connect(config)
    connect_s = time.monotonic() - t0
    transport.send(b"hello")
    reply = transport.recv()
    transport.close()
    result_queue.put({"self_id": config.self_id, "ok": reply == b"world", "reply": reply, "connect_s": connect_s})


def _worker_b(result_queue, backend: str, config: TransportConfig) -> None:
    transport = get_transport(backend)
    transport.connect(config)
    received = transport.recv()
    transport.send(b"world")
    transport.close()
    result_queue.put({"self_id": config.self_id, "ok": received == b"hello", "received": received})


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
        cfg_a, cfg_b = transport_config_pair(args.transport, "A", "B", signaling_url, base_port)

        result_queue = MP_CTX.Queue()
        pa = MP_CTX.Process(target=_worker_a, args=(result_queue, args.transport, cfg_a))
        pb = MP_CTX.Process(target=_worker_b, args=(result_queue, args.transport, cfg_b))
        pb.start()
        pa.start()

        results = [result_queue.get(timeout=60), result_queue.get(timeout=60)]
        pa.join(timeout=10)
        pb.join(timeout=10)
    finally:
        if signaling is not None:
            signaling.stop()

    print(f"=== Test 1: hello/world ({args.transport}) ===")
    ok = True
    for r in results:
        print(f"  {r['self_id']}: {r}")
        ok = ok and r["ok"]

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
