"""Test 3: 1000 ping-pong packets. Measures average/P95/P99 RTT, jitter,
and packet loss. Same worker code for both transports.

Run:
    python3 test3_ping_pong.py --transport tcp
    python3 test3_ping_pong.py --transport udp
"""
from __future__ import annotations

import argparse
import statistics
import struct
import sys
import time

from _common import MP_CTX, SignalingServer, free_port, transport_config_pair

from vllm.transport import TransportConfig, get_transport

N_PINGS = 1000
PING_TIMEOUT_S = 2.0
_HEADER = struct.Struct("!I")  # sequence number


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def _ponger(result_queue, backend: str, config: TransportConfig, n: int) -> None:
    transport = get_transport(backend)
    transport.connect(config)
    echoed = 0
    for _ in range(n):
        try:
            payload = transport.recv(timeout=PING_TIMEOUT_S + 1)
        except TimeoutError:
            break
        transport.send(payload)
        echoed += 1
    transport.close()
    result_queue.put({"self_id": config.self_id, "role": "ponger", "echoed": echoed})


def _pinger(result_queue, backend: str, config: TransportConfig, n: int) -> None:
    transport = get_transport(backend)
    transport.connect(config)
    rtts_ms: list[float] = []
    lost = 0
    for seq in range(n):
        t0 = time.monotonic()
        transport.send(_HEADER.pack(seq))
        try:
            transport.recv(timeout=PING_TIMEOUT_S)
            rtts_ms.append((time.monotonic() - t0) * 1000)
        except TimeoutError:
            lost += 1
    transport.close()

    jitter = (
        sum(abs(rtts_ms[i] - rtts_ms[i - 1]) for i in range(1, len(rtts_ms))) / (len(rtts_ms) - 1)
        if len(rtts_ms) > 1
        else 0.0
    )
    result_queue.put({
        "self_id": config.self_id,
        "role": "pinger",
        "sent": n,
        "received": len(rtts_ms),
        "lost": lost,
        "loss_pct": 100.0 * lost / n,
        "avg_rtt_ms": statistics.mean(rtts_ms) if rtts_ms else float("nan"),
        "p95_rtt_ms": percentile(rtts_ms, 95),
        "p99_rtt_ms": percentile(rtts_ms, 99),
        "jitter_ms": jitter,
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["tcp", "udp"], required=True)
    parser.add_argument("--n", type=int, default=N_PINGS)
    args = parser.parse_args()

    signaling = SignalingServer() if args.transport == "udp" else None
    signaling_url = None
    if signaling is not None:
        signaling.start()
        signaling_url = signaling.url

    try:
        base_port = free_port()
        cfg_pinger, cfg_ponger = transport_config_pair(args.transport, "A", "B", signaling_url, base_port)

        result_queue = MP_CTX.Queue()
        p_pong = MP_CTX.Process(target=_ponger, args=(result_queue, args.transport, cfg_ponger, args.n))
        p_ping = MP_CTX.Process(target=_pinger, args=(result_queue, args.transport, cfg_pinger, args.n))
        p_pong.start()
        p_ping.start()

        results = [result_queue.get(timeout=180), result_queue.get(timeout=180)]
        p_ping.join(timeout=15)
        p_pong.join(timeout=15)
    finally:
        if signaling is not None:
            signaling.stop()

    r = next(x for x in results if x["role"] == "pinger")

    print(f"=== Test 3: {args.n} ping-pong packets ({args.transport}) ===")
    print(f"  avg RTT: {r['avg_rtt_ms']:.3f} ms   P95: {r['p95_rtt_ms']:.3f} ms   P99: {r['p99_rtt_ms']:.3f} ms")
    print(f"  jitter:  {r['jitter_ms']:.3f} ms")
    print(f"  loss:    {r['lost']}/{r['sent']} ({r['loss_pct']:.2f}%)")

    ok = r["received"] > 0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
