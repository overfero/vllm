"""Test 23 (new): fault injection via a lossy-socket shim.

Real loopback almost never drops/reorders/duplicates packets, so none of
the existing tests exercise aioquic's actual retransmission/congestion-
control/reordering-tolerance machinery - only the happy path. This
monkeypatches the `socket.socket` CLASS (not an instance) so every
outbound `sendto()` call on ANY UDP socket the process creates (hole-
punch pings included, not just QUIC traffic - a real lossy link would
drop those too) is independently subject to random drop/duplicate/
reorder-via-jitter before it ever reaches the real kernel socket.

Only `sendto()` is overridden (`_LossySocket` subclasses `socket.socket`
and changes nothing else): this process's own signaling-server HTTP
traffic goes through TCP's `send()`/`sendall()`, never `sendto()`, so it
is completely unaffected - only UDP is ever lossy here.

Because this patches the GLOBAL `socket.socket` class, it is only ever
applied INSIDE the already-forked sender child process's own worker
function body (`MP_CTX` is the `fork` context - see `_common.py` - so
each child already has independent memory by the time its target
function starts running), never in the parent before fork(), or it would
also make the receiver's and the signaling server's sockets lossy.

Run:
    python3 test23_fault_injection_lossy.py --transport quic
"""
from __future__ import annotations

import argparse
import random
import socket
import sys
import threading
import time

from _common import MP_CTX, SignalingServer, free_port, transport_config_pair

from vllm.transport import TransportConfig, get_transport

PAYLOAD_SIZE = 8 * 1024 * 1024  # large enough to span many packets - loss/reorder needs several in flight
DROP_P = 0.05
DUP_P = 0.03
REORDER_P = 0.15
_SEED = 1234


def _make_payload() -> bytes:
    pattern = bytes(random.Random(_SEED).getrandbits(8) for _ in range(64))
    return pattern * (PAYLOAD_SIZE // len(pattern))


_RealSocket = socket.socket  # captured NOW, before _sender() ever patches socket.socket - see below


class _LossySocket(_RealSocket):
    """See module docstring - only sendto() is touched.

    Must call `_RealSocket.sendto(...)`, NOT `socket.socket.sendto(...)`,
    from inside this override: `socket.socket` is a plain module-attribute
    lookup, re-resolved every time it's referenced - by the time
    `_sender()` has patched it to `_LossySocket` itself, writing
    `socket.socket.sendto(self, ...)` here would recurse into THIS SAME
    override forever (a real bug hit while writing this test: it looked
    correct because subclassing captures the base class once for the MRO,
    but a later in-body *name* lookup of `socket.socket` does not reuse
    that - it re-reads the (by-then-patched) module attribute).
    """

    def sendto(self, data, address):  # noqa: D401
        def _real_send() -> None:
            try:
                _RealSocket.sendto(self, data, address)
            except OSError:
                pass  # socket may already be closed by the time a delayed send fires - fine, just drop it

        if random.random() < DROP_P:
            return len(data)  # silently "sent" - simulates a dropped packet, sender gets no error

        delay = random.uniform(0.003, 0.02) if random.random() < REORDER_P else 0.0
        if delay > 0:
            t = threading.Timer(delay, _real_send)
            t.daemon = True
            t.start()
        else:
            _real_send()

        if random.random() < DUP_P:
            t = threading.Timer(delay + random.uniform(0.001, 0.005), _real_send)
            t.daemon = True
            t.start()

        return len(data)


def _sender(result_queue, backend: str, config: TransportConfig) -> None:
    socket.socket = _LossySocket  # see module docstring - only safe here, post-fork, sender-only

    transport = get_transport(backend)
    payload = _make_payload()
    t0 = time.monotonic()
    transport.connect(config)
    connect_s = time.monotonic() - t0
    transport.send(payload)
    ack = transport.recv(timeout=90).decode()
    transport.close()
    result_queue.put({"self_id": config.self_id, "role": "sender", "connect_s": connect_s, "ack": ack})


def _receiver(result_queue, backend: str, config: TransportConfig) -> None:
    transport = get_transport(backend)
    transport.connect(config)
    received = transport.recv(timeout=90)
    byte_perfect = received == _make_payload()
    transport.send(b"OK" if byte_perfect else b"FAIL")
    transport.close()
    result_queue.put({
        "self_id": config.self_id, "role": "receiver",
        "byte_perfect": byte_perfect, "received_len": len(received),
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["tcp", "udp", "quic"], default="quic")
    args = parser.parse_args()
    if args.transport != "quic":
        print(f"NOTE: this test exercises aioquic's own loss-recovery/congestion control - "
              f"running with --transport {args.transport} anyway will exercise that backend's "
              f"OWN reliability layer instead (udp_transport.py's hand-rolled retry logic, or "
              f"TCP's kernel-level retransmission), which is a different (and, for udp, weaker) "
              f"code path than what this test was written to validate.")

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

        results = [result_queue.get(timeout=120), result_queue.get(timeout=120)]
        p_send.join(timeout=15)
        p_recv.join(timeout=15)
    finally:
        if signaling is not None:
            signaling.stop()

    sender = next(r for r in results if r["role"] == "sender")
    receiver = next(r for r in results if r["role"] == "receiver")

    print(f"=== Test 23: fault injection (drop={DROP_P:.0%} dup={DUP_P:.0%} reorder={REORDER_P:.0%}) "
          f"({args.transport}) ===")
    print(f"  connect: {sender['connect_s']:.2f}s")
    print(f"  byte-perfect despite injected loss/dup/reorder: {receiver['byte_perfect']} "
          f"({receiver['received_len']}/{PAYLOAD_SIZE} bytes)")
    print(f"  sender ack: {sender['ack']}")

    ok = receiver["byte_perfect"] and sender["ack"] == "OK"
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
