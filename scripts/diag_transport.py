#!/usr/bin/env python3
"""Standalone, model-free diagnostic for the UDP transport's RPC-control-
channel path (connect + send/recv a realistic-size payload a few times).

Mirrors exactly what stage_server.py's `_open_driver_rpc_link` (listen side)
and rpc_executor.py's `_connect_remote_stages` (connect side) do, minus all
the vLLM engine/model machinery - lets us iterate on the transport in
seconds instead of the minutes a full weight-load cycle takes.

Usage:
    # listener side (like a stage_server.py)
    python3 scripts/diag_transport.py --role listen --self-id A --peer-id C \\
        --signaling-url $SIGNALING_URL --port 40001

    # connector side (like the driver)
    python3 scripts/diag_transport.py --role connect --self-id C --peer-id A \\
        --signaling-url $SIGNALING_URL --port 40001 --rounds 5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vllm.transport import TransportConfig, get_transport  # noqa: E402
from vllm.transport.udp_transport import _AdapterProtocol  # noqa: E402

_orig_handle = _AdapterProtocol._handle_datagram
_orig_send_multi = _AdapterProtocol.send_multi


def _traced_handle(self, data: bytes, addr) -> None:
    tag = data[0:1] if data else b"?"
    print(f"[diag] RECV tag={tag!r} len={len(data)} from={addr} "
          f"recent_before={list(self._recent_peer_addrs)}", file=sys.stderr, flush=True)
    _orig_handle(self, data, addr)


def _traced_send_multi(self, data: bytes) -> None:
    tag = data[0:1] if data else b"?"
    print(f"[diag] SEND_MULTI tag={tag!r} len={len(data)} to={list(self._recent_peer_addrs)}",
          file=sys.stderr, flush=True)
    _orig_send_multi(self, data)


_AdapterProtocol._handle_datagram = _traced_handle
_AdapterProtocol.send_multi = _traced_send_multi


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=["listen", "connect"], required=True)
    p.add_argument("--self-id", required=True)
    p.add_argument("--peer-id", required=True)
    p.add_argument("--signaling-url", required=True)
    p.add_argument("--port", type=int, default=40001)
    p.add_argument("--connect-timeout", type=float, default=60.0)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--payload-size", type=int, default=1302)
    args = p.parse_args()

    transport = get_transport("udp")
    print(f"[diag] connecting role={args.role} self={args.self_id} peer={args.peer_id}",
          file=sys.stderr, flush=True)
    t0 = time.monotonic()
    transport.connect(
        TransportConfig(
            self_id=args.self_id,
            peer_id=args.peer_id,
            host="0.0.0.0" if args.role == "listen" else "127.0.0.1",
            port=args.port,
            listen=(args.role == "listen"),
            signaling_url=args.signaling_url,
            udp_mode="preserve",
            udp_port=args.port,
            connect_timeout=args.connect_timeout,
        )
    )
    print(f"[diag] connected in {time.monotonic() - t0:.1f}s", file=sys.stderr, flush=True)

    if args.role == "listen":
        # Echo loop: receive, print, send back an ack.
        while True:
            try:
                raw = transport.recv(timeout=None)
            except (TimeoutError, ConnectionError, OSError) as e:
                print(f"[diag] listen: transport error: {e}", file=sys.stderr, flush=True)
                return 1
            print(f"[diag] listen: got {len(raw)} bytes, echoing back", file=sys.stderr, flush=True)
            transport.send(b"ack:" + raw[:16])
    else:
        payload = b"x" * args.payload_size
        # Idle gap BEFORE first send, matching the real ~2min gap we saw in
        # the full run, to reproduce the exact failure condition.
        idle_s = 30.0
        print(f"[diag] connect: idling {idle_s}s before first send (reproduce real timing)",
              file=sys.stderr, flush=True)
        time.sleep(idle_s)
        for i in range(args.rounds):
            t0 = time.monotonic()
            try:
                transport.send(payload)
                reply = transport.recv(timeout=15.0)
                print(f"[diag] connect: round {i} OK in {time.monotonic()-t0:.2f}s, "
                      f"reply={reply[:20]!r}", file=sys.stderr, flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[diag] connect: round {i} FAILED after {time.monotonic()-t0:.2f}s: "
                      f"{e!r}", file=sys.stderr, flush=True)
            time.sleep(5.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
