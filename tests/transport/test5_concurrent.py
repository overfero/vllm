"""Test 5: four workers (A, B, C, D), two independent pairs (A<->B, C<->D)
communicating simultaneously. Verifies no deadlock, no corruption (each
side gets exactly its own partner's data, not the other pair's), and no
race conditions. Same worker code for both transports.

For --transport udp this also exercises per-pair signaling isolation: both
pairs register to and poll the *same* signaling server concurrently, and
must each get their own independent synchronized punch time without
interfering with each other.

Run:
    python3 test5_concurrent.py --transport tcp
    python3 test5_concurrent.py --transport udp
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from _common import MP_CTX, SignalingServer, free_port, transport_config_pair

from vllm.transport import TransportConfig, get_transport

PAYLOAD_SIZE = 64 * 1024


def _worker(result_queue, backend: str, config: TransportConfig, role: str) -> None:
    transport = get_transport(backend)
    t0 = time.monotonic()
    transport.connect(config)
    connect_s = time.monotonic() - t0

    my_payload = f"{config.self_id}:".encode() + os.urandom(PAYLOAD_SIZE)
    if role == "initiator":
        transport.send(my_payload)
        their_payload = transport.recv(timeout=30)
    else:
        their_payload = transport.recv(timeout=30)
        transport.send(my_payload)
    transport.close()

    expected_prefix = f"{config.peer_id}:".encode()
    result_queue.put({
        "self_id": config.self_id,
        "peer_id": config.peer_id,
        "connect_s": connect_s,
        "no_cross_talk": their_payload.startswith(expected_prefix),
        "byte_perfect_len": len(their_payload) == PAYLOAD_SIZE + len(expected_prefix),
    })


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
        port_ab = free_port()
        port_cd = free_port() if args.transport == "tcp" else port_ab + 10
        cfg_a, cfg_b = transport_config_pair(args.transport, "A", "B", signaling_url, port_ab)
        cfg_c, cfg_d = transport_config_pair(args.transport, "C", "D", signaling_url, port_cd)

        result_queue = MP_CTX.Queue()
        procs = [
            MP_CTX.Process(target=_worker, args=(result_queue, args.transport, cfg_a, "initiator")),
            MP_CTX.Process(target=_worker, args=(result_queue, args.transport, cfg_b, "responder")),
            MP_CTX.Process(target=_worker, args=(result_queue, args.transport, cfg_c, "initiator")),
            MP_CTX.Process(target=_worker, args=(result_queue, args.transport, cfg_d, "responder")),
        ]
        t_start = time.monotonic()
        for p in procs:  # all four launched together - this is the concurrency under test
            p.start()

        results = [result_queue.get(timeout=60) for _ in procs]
        total_s = time.monotonic() - t_start
        for p in procs:
            p.join(timeout=15)

        deadlocked = any(p.is_alive() for p in procs)
        for p in procs:
            if p.is_alive():
                p.terminate()
    finally:
        if signaling is not None:
            signaling.stop()

    print(f"=== Test 5: concurrent 4-worker communication ({args.transport}) ===")
    print(f"  wall time for all 4 workers: {total_s:.2f} s   deadlock: {deadlocked}")
    ok = not deadlocked
    for r in sorted(results, key=lambda x: x["self_id"]):
        print(
            f"  {r['self_id']} <-> {r['peer_id']}: connect={r['connect_s']:.2f}s  "
            f"no_cross_talk={r['no_cross_talk']}  byte_perfect_len={r['byte_perfect_len']}"
        )
        ok = ok and r["no_cross_talk"] and r["byte_perfect_len"]

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
