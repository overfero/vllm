"""Test 22 (new): large tensor up to 500MB, plus a repeated-transfer
memory-flatness check on ONE connection.

`test7_tensor_large.py` only goes up to 64MB and never sends more than a
handful of messages, so it could never have caught a "completed message
buffer never evicted" bug (exactly the class of bug this project's
UDPTransport had, and QUICTransport's own one-stream-per-message design
had a different variant of - see quic_transport.py's `_on_stream_data`
docstring). This test sends 50 messages of 10MB (500MB cumulative) back
to back on the SAME connection and samples this process's own RSS after
each one: a real "never evicted" leak grows roughly linearly with total
bytes transferred, which is easy to tell apart from normal allocator
noise.

Run:
    python3 test22_tensor_500mb_memory.py --transport quic
"""
from __future__ import annotations

import argparse
import gc
import sys
import time

import psutil
import torch
from _common import MP_CTX, SignalingServer, free_port, transport_config_pair

from vllm.transport import TransportConfig, get_transport
from vllm.transport.tensor import TransportProcessGroup

MB = 1024 * 1024
LARGE_SIZE_BYTES = 500 * MB
REPEAT_MESSAGE_BYTES = 10 * MB
REPEAT_COUNT = 50  # 50 x 10MB = 500MB cumulative, on the SAME connection
WARMUP_SAMPLES = 5  # excluded from the leak comparison (allocator/arena warm-up)


def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / MB


def _sender(result_queue, backend: str, config: TransportConfig) -> None:
    transport = get_transport(backend)
    transport.connect(config)
    pg = TransportProcessGroup(transport)

    big = torch.randn(LARGE_SIZE_BYTES // 4, dtype=torch.float32)
    t0 = time.monotonic()
    pg.send_tensor(big)
    large_ack = transport.recv(timeout=180).decode()
    large_elapsed = time.monotonic() - t0
    del big
    gc.collect()

    rss_samples = []
    small = torch.randn(REPEAT_MESSAGE_BYTES // 4, dtype=torch.float32)
    for _ in range(REPEAT_COUNT):
        pg.send_tensor(small)
        transport.recv(timeout=30)  # per-message ack - also paces sender to receiver
        rss_samples.append(_rss_mb())

    transport.close()
    result_queue.put({
        "self_id": config.self_id, "role": "sender",
        "large_ack": large_ack, "large_elapsed_s": large_elapsed,
        "rss_samples_mb": rss_samples,
    })


def _receiver(result_queue, backend: str, config: TransportConfig) -> None:
    transport = get_transport(backend)
    transport.connect(config)
    pg = TransportProcessGroup(transport)

    big, _ = pg.recv_tensor(timeout=180)
    large_ok = big.numel() == LARGE_SIZE_BYTES // 4 and big.dtype == torch.float32
    transport.send(b"OK" if large_ok else b"FAIL")
    del big
    gc.collect()

    rss_samples = []
    ok_count = 0
    expected_numel = REPEAT_MESSAGE_BYTES // 4
    for _ in range(REPEAT_COUNT):
        tensor, _ = pg.recv_tensor(timeout=30)
        ok_count += tensor.numel() == expected_numel
        transport.send(b"ack")
        del tensor
        rss_samples.append(_rss_mb())

    transport.close()
    result_queue.put({
        "self_id": config.self_id, "role": "receiver",
        "large_ok": large_ok, "repeat_ok_count": ok_count,
        "rss_samples_mb": rss_samples,
    })


def _leak_verdict(rss_samples: list[float], per_message_mb: float) -> tuple[bool, str]:
    """Compares late-window average RSS against an early (post-warmup)
    window. A real "never evicted" leak would grow by roughly
    REPEAT_COUNT * per_message_mb; a generous multiple of ONE message's
    size is used as the noise tolerance - well below what a real leak
    would produce, comfortably above normal GC/allocator jitter.
    """
    if len(rss_samples) <= WARMUP_SAMPLES + 5:
        return True, "too few samples to judge - not a failure"
    early = rss_samples[WARMUP_SAMPLES:WARMUP_SAMPLES + 5]
    late = rss_samples[-5:]
    growth = (sum(late) / len(late)) - (sum(early) / len(early))
    threshold = per_message_mb * 3
    ok = growth < threshold
    return ok, (f"growth={growth:.1f}MB threshold={threshold:.1f}MB "
                f"(early avg={sum(early) / len(early):.1f}MB, late avg={sum(late) / len(late):.1f}MB)")


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

        results = [result_queue.get(timeout=300), result_queue.get(timeout=300)]
        p_send.join(timeout=20)
        p_recv.join(timeout=20)
    finally:
        if signaling is not None:
            signaling.stop()

    sender = next(r for r in results if r["role"] == "sender")
    receiver = next(r for r in results if r["role"] == "receiver")

    print(f"=== Test 22: 500MB tensor + repeated-transfer memory flatness ({args.transport}) ===")
    print(f"  500MB single transfer: ack={sender['large_ack']} ok={receiver['large_ok']} "
          f"elapsed={sender['large_elapsed_s']:.2f}s")

    per_msg_mb = REPEAT_MESSAGE_BYTES / MB
    sender_ok, sender_detail = _leak_verdict(sender["rss_samples_mb"], per_msg_mb)
    receiver_ok, receiver_detail = _leak_verdict(receiver["rss_samples_mb"], per_msg_mb)
    print(f"  sender RSS over {REPEAT_COUNT} x {per_msg_mb:.0f}MB messages:   {sender_detail}")
    print(f"  receiver RSS over {REPEAT_COUNT} x {per_msg_mb:.0f}MB messages: {receiver_detail}")
    print(f"  receiver correctness: {receiver['repeat_ok_count']}/{REPEAT_COUNT}")

    ok = (
        sender["large_ack"] == "OK" and receiver["large_ok"]
        and receiver["repeat_ok_count"] == REPEAT_COUNT
        and sender_ok and receiver_ok
    )
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
