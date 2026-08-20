"""Test 26 (new, QUIC broker): several named channels multiplexed over
ONE real QUIC connection (`quic_broker.py`), instead of each channel
hole-punching/handshaking its own separate connection the way plain
`QUICTransport` does. This is what this project's real topology actually
needs (TP=2 PP link + 1 RPC control channel between the same two
machines) - see `quic_broker.py`'s module docstring for the full
rationale.

Verifies, end to end through the real local Unix-socket IPC hop
(`QuicMultiplexedTransport`), not just the in-process broker API:
  1. Each channel's messages are delivered to the matching channel on the
     peer side, and ONLY that channel - no cross-channel leakage.
  2. Per-channel message ORDER is preserved (matches quic_transport.py's
     own single-ordered-stream-per-direction guarantee, now per channel).
  3. All channels genuinely share one underlying QUIC connection (checked
     directly: each side's `QuicBroker` has exactly one `_quic`
     QuicConnection object backing every channel).
  4. Every message survives a close() immediately after the last send -
     see `QuicBroker._close_async`'s docstring for the real, qlog-found
     premature-close bug this guards against (closing right after the
     local work finishes, while some data was still only optimistically
     "sent once" and not yet actually acknowledged by the peer).

Run:
    python3 test26_quic_broker_multiplexing.py
"""
from __future__ import annotations

import sys
import threading

from _common import MP_CTX, SignalingServer, free_port

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from vllm.transport import TransportConfig  # noqa: E402
from vllm.transport.quic_broker import QuicBroker, broker_socket_path, serve_local_broker  # noqa: E402
from vllm.transport.quic_multiplexed_transport import QuicMultiplexedTransport  # noqa: E402

CHANNELS = ["tp0", "tp1", "rpc"]
MESSAGES_PER_CHANNEL = 20


def _channel_worker(self_id: str, peer_id: str, channel: str, results: dict, errors: list) -> None:
    """Runs as a thread inside the machine-side process: connects to the
    LOCAL broker (already running in this same process) for `channel`,
    sends `MESSAGES_PER_CHANNEL` distinguishable messages, and receives
    the peer's matching stream of messages back - proves both directions
    route correctly for this one channel while 2 other channels are doing
    the exact same thing concurrently over the SAME QUIC connection."""
    try:
        transport = QuicMultiplexedTransport()
        transport.connect(
            TransportConfig(
                self_id=f"{self_id}-{channel}",
                peer_id=f"peer-{channel}",
                connect_timeout=30.0,
                extra={"broker_socket_path": broker_socket_path(self_id, peer_id), "channel": channel},
            )
        )
        sent = []
        for i in range(MESSAGES_PER_CHANNEL):
            msg = f"{channel}:{i}".encode()
            transport.send(msg)
            sent.append(msg)

        received = []
        for _ in range(MESSAGES_PER_CHANNEL):
            received.append(transport.recv(timeout=15.0))

        transport.close()
        results[channel] = {"sent": sent, "received": received}
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{channel}: {type(exc).__name__}: {exc}")


def _machine_side(result_queue, self_id: str, peer_id: str, signaling_url: str, udp_port: int, listen: bool) -> None:
    broker = QuicBroker()
    broker.connect(
        self_id=self_id, peer_id=peer_id, signaling_url=signaling_url,
        udp_port=udp_port, listen=listen, connect_timeout=60.0,
    )
    serve_local_broker(broker, broker_socket_path(self_id, peer_id))

    results: dict[str, dict] = {}
    errors: list[str] = []
    threads = [
        threading.Thread(target=_channel_worker, args=(self_id, peer_id, ch, results, errors))
        for ch in CHANNELS
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    quic_conn_id = id(broker._quic)
    broker.close()

    result_queue.put({
        "self_id": self_id,
        "results": results,
        "errors": errors,
        "quic_conn_id": quic_conn_id,
        "alive_threads": [t.name for t in threads if t.is_alive()],
    })


def main() -> int:
    signaling = SignalingServer()
    signaling.start()
    try:
        base_port = free_port()
        result_queue = MP_CTX.Queue()
        p_a = MP_CTX.Process(
            target=_machine_side,
            args=(result_queue, "MachineA", "MachineB", signaling.url, base_port, True),
        )
        p_b = MP_CTX.Process(
            target=_machine_side,
            args=(result_queue, "MachineB", "MachineA", signaling.url, base_port + 1, False),
        )
        p_a.start()
        p_b.start()

        results = [result_queue.get(timeout=90), result_queue.get(timeout=90)]
        p_a.join(timeout=15)
        p_b.join(timeout=15)
    finally:
        signaling.stop()

    a = next(r for r in results if r["self_id"] == "MachineA")
    b = next(r for r in results if r["self_id"] == "MachineB")

    print("=== Test 26: multiple channels multiplexed over one QUIC connection ===")

    ok = True
    for side_name, side in (("MachineA", a), ("MachineB", b)):
        if side["errors"]:
            print(f"  {side_name}: ERRORS: {side['errors']}")
            ok = False
        if side["alive_threads"]:
            print(f"  {side_name}: threads still alive (timed out): {side['alive_threads']}")
            ok = False
        print(f"  {side_name}: single QuicConnection object backs all channels "
              f"(id={side['quic_conn_id']})")

    for channel in CHANNELS:
        a_sent = a["results"].get(channel, {}).get("sent")
        a_recv = a["results"].get(channel, {}).get("received")
        b_sent = b["results"].get(channel, {}).get("sent")
        b_recv = b["results"].get(channel, {}).get("received")
        if a_sent is None or b_sent is None:
            print(f"  channel {channel!r}: missing results (see errors above)")
            ok = False
            continue
        order_ok = a_recv == b_sent and b_recv == a_sent
        no_cross_talk = all(msg.startswith(f"{channel}:".encode()) for msg in a_recv + b_recv)
        print(f"  channel {channel!r}: order_preserved={order_ok} no_cross_channel_leakage={no_cross_talk} "
              f"({len(a_recv)}/{MESSAGES_PER_CHANNEL} msgs each direction)")
        ok = ok and order_ok and no_cross_talk

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
