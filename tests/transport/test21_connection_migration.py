"""Test 21 (new, QUIC-only): connection migration mid-transfer.

The whole reason this project picked QUIC over plain TCP hole punching
(see the earlier TCP-feasibility check in project history) is that TCP
dies outright on any 4-tuple change, while QUIC survives a NAT remapping
its outbound port via connection migration (RFC 9000 SS9,
PATH_CHALLENGE/PATH_RESPONSE) - completely transparent to the
application. This had NO test coverage before (gap called out explicitly
in the migration spec's SS5.1) despite being the headline feature.

Real NAT rebinding can't be triggered on demand from a test harness, so
this uses `QUICTransport.simulate_rebind()` - a test-only hook that does
the one thing a test CAN control: actually close and reopen this
process's own UDP socket (the only way to really change a socket's
source port on Linux), then keep driving the SAME `QuicConnection` object
over it. From the PEER's perspective this is indistinguishable from a
real NAT remap: packets for an existing connection ID suddenly arrive
from a new source address, and aioquic's own (unmodified) migration logic
has to validate and switch to it.

Run:
    python3 test21_connection_migration.py --transport quic
"""
from __future__ import annotations

import argparse
import sys
import time

from _common import MP_CTX, SignalingServer, free_port, transport_config_pair

from vllm.transport import TransportConfig, get_transport

PAYLOAD_SIZE = 20 * 1024 * 1024  # large enough that migration happens well before the transfer drains


def _sender(result_queue, backend: str, config: TransportConfig) -> None:
    transport = get_transport(backend)
    transport.connect(config)

    old_sockname = transport._transport_obj.get_extra_info("sockname")
    payload = b"M" * PAYLOAD_SIZE
    t0 = time.monotonic()
    transport.send(payload)  # returns once queued - most of a 20MB payload is still in flight at this point
    transport.simulate_rebind(new_local_port=0)  # actually close+reopen the local socket, mid-transfer
    new_sockname = transport._transport_obj.get_extra_info("sockname")

    ack = transport.recv(timeout=60)
    elapsed = time.monotonic() - t0
    transport.close()
    result_queue.put({
        "self_id": config.self_id,
        "role": "sender",
        "old_port": old_sockname[1],
        "new_port": new_sockname[1],
        "ack": ack.decode(),
        "elapsed_s": elapsed,
    })


def _receiver(result_queue, backend: str, config: TransportConfig) -> None:
    transport = get_transport(backend)
    transport.connect(config)
    received = transport.recv(timeout=60)
    byte_perfect = received == b"M" * PAYLOAD_SIZE
    rebind_events = transport._protocol.rebind_event_count
    transport.send(b"OK" if byte_perfect else b"FAIL")
    transport.close()
    result_queue.put({
        "self_id": config.self_id,
        "role": "receiver",
        "byte_perfect": byte_perfect,
        "received_len": len(received),
        "rebind_events_observed": rebind_events,
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["tcp", "udp", "quic"], default="quic")
    args = parser.parse_args()
    if args.transport != "quic":
        print(f"NOTE: connection migration is QUIC's headline feature - running with "
              f"--transport {args.transport} anyway, but only 'quic' actually migrates "
              f"the socket mid-connection via simulate_rebind(); other backends will "
              f"simply fail this test (that's expected, not a bug in them).")

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

        results = [result_queue.get(timeout=90), result_queue.get(timeout=90)]
        p_send.join(timeout=15)
        p_recv.join(timeout=15)
    finally:
        if signaling is not None:
            signaling.stop()

    sender = next(r for r in results if r["role"] == "sender")
    receiver = next(r for r in results if r["role"] == "receiver")

    print(f"=== Test 21: connection migration mid-transfer ({args.transport}) ===")
    print(f"  sender local port:  {sender['old_port']} -> {sender['new_port']} "
          f"({'changed' if sender['old_port'] != sender['new_port'] else 'UNCHANGED - rebind did not happen!'})")
    print(f"  receiver saw rebind events: {receiver['rebind_events_observed']}")
    print(f"  payload byte-perfect after migration: {receiver['byte_perfect']} "
          f"({receiver['received_len']}/{PAYLOAD_SIZE} bytes)")
    print(f"  elapsed: {sender['elapsed_s']:.2f}s")

    ok = (
        sender["old_port"] != sender["new_port"]
        and receiver["rebind_events_observed"] > 0
        and receiver["byte_perfect"]
        and sender["ack"] == "OK"
    )
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
