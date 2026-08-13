"""Real cross-machine ping-pong latency test using the SAME transport code
path (vllm.transport.udp_transport, real UDP hole punch via the real public
signaling server) the actual PP deployment uses - unlike
tests/transport/test3_ping_pong.py, which runs both sides as local
multiprocessing.Process on one machine (loopback, not representative of
real cross-machine RTT). Each machine runs this script independently with
its own --role.

Usage (run simultaneously on each machine, within seconds of each other -
same lesson as the real deployment: don't wait for one side to be "ready"
before starting the other):
    # Machine A:
    python3 real_ping_pong.py --role pinger --self-name MachineA --peer-name MachineB \
        --signaling-url https://... --udp-port 37001 --n 200
    # Machine B:
    python3 real_ping_pong.py --role ponger --self-name MachineB --peer-name MachineA \
        --signaling-url https://... --udp-port 37002 --n 200
"""
import argparse
import statistics
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm.transport import TransportConfig, get_transport  # noqa: E402

_HEADER = struct.Struct("!I")


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def run_ponger(cfg: TransportConfig, n: int) -> None:
    transport = get_transport("udp")
    print(f"[{cfg.self_id}] connecting to {cfg.peer_id} via {cfg.signaling_url} ...", flush=True)
    transport.connect(cfg)
    print(f"[{cfg.self_id}] connected, echoing {n} packets...", flush=True)
    echoed = 0
    for _ in range(n):
        try:
            payload = transport.recv(timeout=10.0)
        except TimeoutError:
            break
        transport.send(payload)
        echoed += 1
    transport.close()
    print(f"[{cfg.self_id}] done, echoed {echoed}/{n}", flush=True)


def run_pinger(cfg: TransportConfig, n: int) -> None:
    transport = get_transport("udp")
    print(f"[{cfg.self_id}] connecting to {cfg.peer_id} via {cfg.signaling_url} ...", flush=True)
    transport.connect(cfg)
    print(f"[{cfg.self_id}] connected, pinging {n} packets...", flush=True)
    rtts_ms: list[float] = []
    lost = 0
    for seq in range(n):
        t0 = time.monotonic()
        transport.send(_HEADER.pack(seq))
        try:
            transport.recv(timeout=10.0)
            rtts_ms.append((time.monotonic() - t0) * 1000)
        except TimeoutError:
            lost += 1
    transport.close()

    jitter = (
        sum(abs(rtts_ms[i] - rtts_ms[i - 1]) for i in range(1, len(rtts_ms))) / (len(rtts_ms) - 1)
        if len(rtts_ms) > 1
        else 0.0
    )
    print(f"\n=== REAL cross-machine ping-pong: {cfg.self_id} <-> {cfg.peer_id} ({len(rtts_ms)}/{n} received) ===")
    if rtts_ms:
        print(f"  avg RTT: {statistics.mean(rtts_ms):.2f} ms   "
              f"min: {min(rtts_ms):.2f} ms   max: {max(rtts_ms):.2f} ms")
        print(f"  P50: {percentile(rtts_ms, 50):.2f} ms   "
              f"P95: {percentile(rtts_ms, 95):.2f} ms   P99: {percentile(rtts_ms, 99):.2f} ms")
        print(f"  jitter: {jitter:.2f} ms")
    print(f"  loss: {lost}/{n} ({100.0 * lost / n:.2f}%)")
    print(f"PING_PONG_RESULT: self={cfg.self_id} peer={cfg.peer_id} "
          f"avg_ms={statistics.mean(rtts_ms) if rtts_ms else float('nan'):.3f} "
          f"p50_ms={percentile(rtts_ms, 50):.3f} p95_ms={percentile(rtts_ms, 95):.3f} "
          f"loss_pct={100.0 * lost / n:.2f}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["pinger", "ponger"], required=True)
    ap.add_argument("--self-name", required=True)
    ap.add_argument("--peer-name", required=True)
    ap.add_argument("--signaling-url", required=True)
    ap.add_argument("--udp-port", type=int, required=True)
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    cfg = TransportConfig(
        self_id=args.self_name,
        peer_id=args.peer_name,
        signaling_url=args.signaling_url,
        udp_mode="preserve",
        udp_port=args.udp_port,
        connect_timeout=60.0,
    )

    if args.role == "ponger":
        run_ponger(cfg, args.n)
    else:
        run_pinger(cfg, args.n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
