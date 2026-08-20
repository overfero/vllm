"""Standalone entry point for one `QuicBroker` connection to one peer
machine - see `quic_broker.py`'s module docstring for the full rationale
(one shared QUIC connection multiplexing several channels, instead of
each channel hole-punching its own).

Why this needs to be its own OS process (not just a thread started inside
`stage_server.py`/`launch_pp_stage.py`): `launch_pp_stage.py` (the driver
launcher) execs directly into `vllm serve` via `os.execvpe` - that
replaces the ENTIRE process image, including every thread, so a broker
started as a thread there would be destroyed the instant the driver
process execs. Running the broker as a genuinely separate, detached
subprocess (see both launchers' `_maybe_start_quic_broker_daemon` helper)
means it survives that exec and keeps running for the whole deployment's
lifetime, independent of whatever the driver/stage process does.

Usage:
    python3 -m vllm.transport.quic_broker_daemon \\
        --self-id Stage0 --peer-id Stage1 \\
        --signaling-url https://... --udp-port 30100 --listen
"""
from __future__ import annotations

import argparse
import signal
import sys
import threading

from vllm.logger import init_logger
from vllm.transport.quic_broker import QuicBroker, broker_socket_path, serve_local_broker

logger = init_logger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-id", required=True)
    p.add_argument("--peer-id", required=True)
    p.add_argument("--signaling-url", required=True)
    p.add_argument("--udp-port", type=int, required=True)
    p.add_argument("--listen", action="store_true", help="QUIC server role (presents TLS cert)")
    p.add_argument("--connect-timeout", type=float, default=900.0)
    p.add_argument("--ready-file", default=None,
                    help="if set, this file is created once the local Unix-socket listener "
                         "is accepting connections - lets the parent process's own retry "
                         "loop (or a preflight check) confirm the daemon is actually up "
                         "without guessing a sleep duration")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    broker = QuicBroker()
    print(f"[quic_broker_daemon] self={args.self_id} peer={args.peer_id} "
          f"connecting (hole-punch + QUIC handshake)...", file=sys.stderr, flush=True)
    broker.connect(
        self_id=args.self_id,
        peer_id=args.peer_id,
        signaling_url=args.signaling_url,
        udp_port=args.udp_port,
        listen=args.listen,
        connect_timeout=args.connect_timeout,
    )
    print(f"[quic_broker_daemon] self={args.self_id} peer={args.peer_id} "
          f"QUIC connection established - starting local channel listener", file=sys.stderr, flush=True)

    socket_path = broker_socket_path(args.self_id, args.peer_id)
    serve_local_broker(broker, socket_path)

    if args.ready_file:
        with open(args.ready_file, "w") as f:
            f.write(socket_path)
    print(f"[quic_broker_daemon] ready - local channels served at {socket_path}",
          file=sys.stderr, flush=True)

    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    stop.wait()
    print(f"[quic_broker_daemon] self={args.self_id} shutting down", file=sys.stderr, flush=True)
    broker.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
