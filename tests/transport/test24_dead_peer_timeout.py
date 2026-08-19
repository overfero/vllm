"""Test 24 (new): dead peer, killed with SIGKILL mid-transfer, is detected
in bounded time - not a permanent hang.

Direct verification of bug #3 from the migration spec's audit (SS2):
`UDPTransport`'s keepalive never checked for a pong reply, and
`transport.recv()` in `parallel_state.py` is called with no timeout, so a
dead peer meant a permanently hung worker. `QUICTransport` is supposed to
fix this structurally via the QUIC connection's own idle timeout, which
fires `ConnectionTerminated` and unblocks every current/future recv()
with a real exception (see quic_transport.py's `_on_terminated`) - this
test is the only place that claim gets checked against a REAL `kill -9`,
not just a graceful `close()` (which every other test already exercises
constantly and would never catch this class of bug).

Run:
    python3 test24_dead_peer_timeout.py --transport quic
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time

from _common import MP_CTX, SignalingServer, free_port, transport_config_pair

from vllm.transport import TransportConfig, get_transport

# Deliberately short so the test doesn't have to wait out a real default
# idle_timeout (45s) - passed through via TransportConfig, same field a
# real deployment would tune (VLLM_TRANSPORT_QUIC_IDLE_TIMEOUT_S / the env
# var quic_transport.py reads - see that module - or directly here via the
# TransportConfig field, whichever the caller already has in scope).
IDLE_TIMEOUT_S = 5.0
# How long the "victim" process should look alive (sent its first message,
# then goes silent - simulating a peer that hangs/crashes mid-session,
# not one that dies before ever saying anything) before being killed.
ALIVE_S = 1.0


def _victim(ready_queue, backend: str, config: TransportConfig) -> None:
    """Connects, sends one message so the survivor knows the link is up,
    then blocks doing nothing until killed - simulating a peer that
    crashes/hangs mid-session (not a startup failure, which is a
    different, already-covered failure mode - see test1's connect_timeout
    handling)."""
    transport = get_transport(backend)
    transport.connect(config)
    transport.send(b"alive")
    ready_queue.put("sent")
    time.sleep(3600)  # killed by the parent long before this would return


def _survivor(result_queue, backend: str, config: TransportConfig) -> None:
    transport = get_transport(backend)
    transport.connect(config)
    first = transport.recv(timeout=30)  # the victim's one "alive" message

    t0 = time.monotonic()
    try:
        transport.recv(timeout=60)  # victim is dead by now (never sends again) - should raise, not hang
        outcome = "recv() returned instead of raising - BUG"
    except TimeoutError:
        outcome = "TimeoutError (recv()'s OWN timeout fired, not dead-peer detection - see SS3.6 of the spec)"
    except ConnectionError as exc:
        outcome = f"ConnectionError (dead-peer detected): {exc}"
    except Exception as exc:  # noqa: BLE001 - report whatever actually happened, don't hide it
        outcome = f"unexpected {type(exc).__name__}: {exc}"
    detected_s = time.monotonic() - t0

    transport.close()
    result_queue.put({
        "self_id": config.self_id,
        "first_message_ok": first == b"alive",
        "outcome": outcome,
        "detected_s": detected_s,
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["tcp", "udp", "quic"], default="quic")
    args = parser.parse_args()

    signaling = SignalingServer() if args.transport in ("udp", "quic") else None
    signaling_url = None
    if signaling is not None:
        signaling.start()
        signaling_url = signaling.url

    try:
        base_port = free_port()
        cfg_victim, cfg_survivor = transport_config_pair(args.transport, "A", "B", signaling_url, base_port)
        if args.transport == "quic":
            cfg_victim.quic_idle_timeout = IDLE_TIMEOUT_S
            cfg_survivor.quic_idle_timeout = IDLE_TIMEOUT_S

        ready_queue = MP_CTX.Queue()
        result_queue = MP_CTX.Queue()
        p_victim = MP_CTX.Process(target=_victim, args=(ready_queue, args.transport, cfg_victim))
        p_survivor = MP_CTX.Process(target=_survivor, args=(result_queue, args.transport, cfg_survivor))
        p_survivor.start()
        p_victim.start()

        ready_queue.get(timeout=30)  # victim has sent its one message and is now looping forever
        time.sleep(ALIVE_S)

        os.kill(p_victim.pid, signal.SIGKILL)  # the real thing this test is about - no graceful close()
        p_victim.join(timeout=10)

        result = result_queue.get(timeout=90)
        p_survivor.join(timeout=15)
        if p_survivor.is_alive():
            p_survivor.terminate()
    finally:
        if signaling is not None:
            signaling.stop()

    print(f"=== Test 24: dead peer (SIGKILL mid-session) detection ({args.transport}) ===")
    print(f"  first message ok:      {result['first_message_ok']}")
    print(f"  outcome:                {result['outcome']}")
    print(f"  detected after:         {result['detected_s']:.2f}s "
          f"(idle_timeout configured: {IDLE_TIMEOUT_S}s)")

    ok = (
        result["first_message_ok"]
        and result["outcome"].startswith("ConnectionError")
        # bounded, and roughly consistent with the configured idle_timeout - not "eventually", not the
        # unrelated 60s recv() timeout also in play here (that would show up as the TimeoutError branch)
        and result["detected_s"] < IDLE_TIMEOUT_S + 15.0
    )
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
