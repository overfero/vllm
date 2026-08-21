"""Tensor phase / Test 2: large tensors (1/4/16/64 MB), float32. Measures
serialize / transport / deserialize / total-latency timing breakdown,
bandwidth, and CPU utilization for both transports.

Run:
    python3 test7_tensor_large.py --transport tcp
    python3 test7_tensor_large.py --transport udp
"""
from __future__ import annotations

import argparse
import resource
import sys
import time

import torch
from _common import MP_CTX, SignalingServer, free_port, transport_config_pair

from vllm.transport import TransportConfig, get_transport
from vllm.transport.tensor import TransportProcessGroup

MB = 1024 * 1024
SIZES_BYTES = [1 * MB, 4 * MB, 16 * MB, 64 * MB]  # float32 -> 4 bytes/element
ELEMENT_BYTES = 4


def _cpu_pct(cpu_before: tuple[float, float], elapsed_s: float) -> dict:
    after = resource.getrusage(resource.RUSAGE_SELF)
    user_s = after.ru_utime - cpu_before[0]
    sys_s = after.ru_stime - cpu_before[1]
    denom = elapsed_s if elapsed_s > 0 else 1.0
    return {"user_pct": 100 * user_s / denom, "system_pct": 100 * sys_s / denom}


def _cpu_now() -> tuple[float, float]:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return (r.ru_utime, r.ru_stime)


def _sender(result_queue, backend: str, config: TransportConfig, sizes: list[int]) -> None:
    transport = get_transport(backend)
    transport.connect(config)
    pg = TransportProcessGroup(transport)
    per_size = []
    for nbytes in sizes:
        n_elems = nbytes // ELEMENT_BYTES
        tensor = torch.randn(n_elems, dtype=torch.float32)

        cpu_before = _cpu_now()
        t0 = time.monotonic()
        send_stats = pg.send_tensor(tensor)
        ack = transport.recv(timeout=60)  # receiver's "OK"/"FAIL" + it saw N bytes
        total_s = time.monotonic() - t0
        cpu = _cpu_pct(cpu_before, total_s)

        per_size.append({
            "nbytes": nbytes,
            "serialize_s": send_stats.serialize_s,
            "transport_s": send_stats.transport_s,
            "total_latency_s": total_s,
            "bandwidth_mbps": (nbytes * 8) / (total_s * 1_000_000) if total_s > 0 else 0.0,
            "cpu": cpu,
            "ack": ack.decode(),
        })
    transport.close()
    result_queue.put({"self_id": config.self_id, "role": "sender", "results": per_size})


def _receiver(result_queue, backend: str, config: TransportConfig, sizes: list[int]) -> None:
    transport = get_transport(backend)
    transport.connect(config)
    pg = TransportProcessGroup(transport)
    per_size = []
    for nbytes in sizes:
        expected_n = nbytes // ELEMENT_BYTES
        tensor, recv_stats = pg.recv_tensor(timeout=60)
        ok = tensor.numel() == expected_n and tensor.dtype == torch.float32
        transport.send(f"{'OK' if ok else 'FAIL'}:{tensor.numel() * ELEMENT_BYTES}".encode())
        per_size.append({
            "nbytes": nbytes,
            "transport_s": recv_stats.transport_s,
            "deserialize_s": recv_stats.deserialize_s,
            "ok": ok,
        })
    transport.close()
    result_queue.put({"self_id": config.self_id, "role": "receiver", "results": per_size})


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
        p_recv = MP_CTX.Process(target=_receiver, args=(result_queue, args.transport, cfg_receiver, SIZES_BYTES))
        p_send = MP_CTX.Process(target=_sender, args=(result_queue, args.transport, cfg_sender, SIZES_BYTES))
        p_recv.start()
        p_send.start()

        results = [result_queue.get(timeout=180), result_queue.get(timeout=180)]
        p_send.join(timeout=15)
        p_recv.join(timeout=15)
    finally:
        if signaling is not None:
            signaling.stop()

    sender_result = next(r for r in results if r["role"] == "sender")
    receiver_result = next(r for r in results if r["role"] == "receiver")

    print(f"=== Tensor Test 2: large tensors ({args.transport}) ===")
    ok = True
    for s, r in zip(sender_result["results"], receiver_result["results"]):
        mb = s["nbytes"] // MB
        print(
            f"  {mb:3d} MB: serialize={s['serialize_s'] * 1000:7.2f}ms  "
            f"transport(send)={s['transport_s'] * 1000:8.2f}ms  "
            f"transport(recv)={r['transport_s'] * 1000:8.2f}ms  "
            f"deserialize={r['deserialize_s'] * 1000:7.2f}ms  "
            f"total={s['total_latency_s'] * 1000:8.2f}ms  "
            f"bw={s['bandwidth_mbps']:8.2f}Mbps  "
            f"cpu(user/sys)={s['cpu']['user_pct']:.1f}%/{s['cpu']['system_pct']:.1f}%"
        )
        ok = ok and r["ok"] and s["ack"].startswith("OK")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
