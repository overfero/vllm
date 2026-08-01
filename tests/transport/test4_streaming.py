"""Test 4: 100MB streaming transfer. Measures throughput, completion time,
CPU usage, and TCP retransmissions (best-effort, TCP only). Same worker
code for both transports - message chunking is handled uniformly by
send()/recv() on a fixed message size regardless of backend.

Run:
    python3 test4_streaming.py --transport tcp
    python3 test4_streaming.py --transport udp
"""
from __future__ import annotations

import argparse
import os
import resource
import sys
import time

from _common import MP_CTX, SignalingServer, free_port, transport_config_pair

from vllm.transport import TransportConfig, get_transport

MB = 1024 * 1024
TOTAL_BYTES = 100 * MB
MESSAGE_SIZE = 1 * MB  # send() is called TOTAL_BYTES // MESSAGE_SIZE times


def _read_tcp_retrans_segs() -> int | None:
    """System-wide TCP retransmit counter from /proc/net/snmp. Coarser than
    per-socket (no portable per-socket API across kernel versions without
    fragile TCP_INFO struct parsing), but a legitimate before/after delta on
    an otherwise-idle host during the test window."""
    try:
        with open("/proc/net/snmp") as f:
            lines = f.read().splitlines()
        for i, line in enumerate(lines):
            if line.startswith("Tcp:") and "RetransSegs" in line:
                headers = line.split()
                values = lines[i + 1].split()
                return int(values[headers.index("RetransSegs")])
    except (OSError, ValueError, StopIteration):
        return None
    return None


def _cpu_pct(cpu_before: tuple[float, float], elapsed_s: float) -> dict:
    after = resource.getrusage(resource.RUSAGE_SELF)
    user_s = after.ru_utime - cpu_before[0]
    sys_s = after.ru_stime - cpu_before[1]
    denom = elapsed_s if elapsed_s > 0 else 1.0
    return {"user_pct": 100 * user_s / denom, "system_pct": 100 * sys_s / denom, "overall_pct": 100 * (user_s + sys_s) / denom}


def _sender(result_queue, backend: str, config: TransportConfig, total_bytes: int, message_size: int) -> None:
    transport = get_transport(backend)
    transport.connect(config)

    retrans_before = _read_tcp_retrans_segs() if backend == "tcp" else None
    cpu_before = (resource.getrusage(resource.RUSAGE_SELF).ru_utime, resource.getrusage(resource.RUSAGE_SELF).ru_stime)

    chunk = os.urandom(message_size)
    n_messages = total_bytes // message_size
    t0 = time.monotonic()
    for _ in range(n_messages):
        transport.send(chunk)
    ack = transport.recv()  # wait for receiver's final byte-count confirmation
    elapsed = time.monotonic() - t0

    retrans_after = _read_tcp_retrans_segs() if backend == "tcp" else None
    cpu = _cpu_pct(cpu_before, elapsed)
    transport.close()

    retrans_delta = (retrans_after - retrans_before) if retrans_before is not None and retrans_after is not None else None
    result_queue.put({
        "self_id": config.self_id,
        "role": "sender",
        "elapsed_s": elapsed,
        "throughput_mbps": (n_messages * message_size * 8) / (elapsed * 1_000_000),
        "cpu": cpu,
        "tcp_retrans_segs_delta": retrans_delta,
        "peer_confirmed_bytes": int(ack.decode()),
    })


def _receiver(result_queue, backend: str, config: TransportConfig, total_bytes: int, message_size: int) -> None:
    transport = get_transport(backend)
    transport.connect(config)
    n_messages = total_bytes // message_size
    received_bytes = 0
    for _ in range(n_messages):
        payload = transport.recv(timeout=60)
        received_bytes += len(payload)
    transport.send(str(received_bytes).encode())
    transport.close()
    result_queue.put({"self_id": config.self_id, "role": "receiver", "received_bytes": received_bytes})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["tcp", "udp"], required=True)
    parser.add_argument("--total-mb", type=int, default=TOTAL_BYTES // MB)
    args = parser.parse_args()
    total_bytes = args.total_mb * MB

    signaling = SignalingServer() if args.transport == "udp" else None
    signaling_url = None
    if signaling is not None:
        signaling.start()
        signaling_url = signaling.url

    try:
        base_port = free_port()
        cfg_sender, cfg_receiver = transport_config_pair(args.transport, "A", "B", signaling_url, base_port)

        result_queue = MP_CTX.Queue()
        p_recv = MP_CTX.Process(
            target=_receiver, args=(result_queue, args.transport, cfg_receiver, total_bytes, MESSAGE_SIZE)
        )
        p_send = MP_CTX.Process(
            target=_sender, args=(result_queue, args.transport, cfg_sender, total_bytes, MESSAGE_SIZE)
        )
        p_recv.start()
        p_send.start()

        results = [result_queue.get(timeout=300), result_queue.get(timeout=300)]
        p_send.join(timeout=20)
        p_recv.join(timeout=20)
    finally:
        if signaling is not None:
            signaling.stop()

    sender = next(r for r in results if r["role"] == "sender")
    receiver = next(r for r in results if r["role"] == "receiver")

    print(f"=== Test 4: {args.total_mb} MB streaming transfer ({args.transport}) ===")
    print(f"  completion time: {sender['elapsed_s']:.2f} s")
    print(f"  throughput:      {sender['throughput_mbps']:.2f} Mbps")
    print(
        f"  CPU (sender):    user={sender['cpu']['user_pct']:.1f}%  "
        f"system={sender['cpu']['system_pct']:.1f}%  overall={sender['cpu']['overall_pct']:.1f}%"
    )
    retrans = sender["tcp_retrans_segs_delta"]
    print(f"  TCP retransmits: {retrans if retrans is not None else 'N/A (udp, or /proc/net/snmp unavailable)'}")
    print(f"  bytes confirmed by receiver: {receiver['received_bytes']} / {total_bytes}")

    ok = receiver["received_bytes"] == total_bytes
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
