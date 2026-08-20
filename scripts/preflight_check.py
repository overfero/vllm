#!/usr/bin/env python3
"""Pre-flight transport connectivity check for one pipeline stage - run
this BEFORE `launch_pp_stage.py` on each machine, so a NAT/signaling/
firewall problem is caught in seconds instead of after a 10+ minute model
load. Exits non-zero with a specific, actionable message on failure; only
prints "PREFLIGHT OK" and exits 0 if every expected link actually
connects.

This intentionally duplicates none of the transport internals - it calls
the exact same `establish_pp_transports` helper `TransportPPWorker` uses
for real, just standalone and with `local_rank=0` for both of a machine's
GPUs' worth of links (a real per-GPU-process preflight would need to run
once per local GPU; this script checks local_rank 0 and 1 by default,
covering the TP=2 target cluster, since a NAT/signaling failure affects
both local ranks identically - only the port differs).

Usage (see README_RUN_GPTOSS_CLUSTER.md for the exact 3-machine
invocations):

    python3 scripts/preflight_check.py \\
        --pp-rank 1 --pp-world-size 3 \\
        --self-name MachineB --prev-name MachineA --next-name MachineC \\
        --transport udp --signaling-url $SIGNALING_URL \\
        --local-ranks 0,1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pp-rank", type=int, required=True)
    p.add_argument("--pp-world-size", type=int, required=True)
    p.add_argument("--self-name", required=True)
    p.add_argument("--prev-name", default=None)
    p.add_argument("--next-name", default=None)
    p.add_argument("--transport", choices=["tcp", "udp", "quic", "quic-shared"], default="udp")
    p.add_argument("--signaling-url", default=None)
    p.add_argument("--udp-port-base", type=int, default=30000)
    p.add_argument("--tcp-port-base", type=int, default=30000)
    p.add_argument("--tcp-connect-host-prev", default=None)
    p.add_argument("--tcp-connect-host-next", default=None)
    p.add_argument("--local-ranks", default="0,1", help="comma-separated local GPU ranks to check (default: 0,1 for TP=2)")
    p.add_argument("--timeout", type=float, default=30.0, help="per-link connect timeout, seconds")
    return p


def check_signaling_server(signaling_url: str, timeout: float) -> tuple[bool, str]:
    import urllib.error
    import urllib.request

    # Real contract (udp_holepunch/signaling_server.py): GET /peer/{peer_id}
    # requires a `self_id` query param and 404s if peer_id isn't registered
    # yet - that 404 is the expected, healthy response for an unregistered
    # probe id, not a failure.
    try:
        req = urllib.request.Request(
            f"{signaling_url.rstrip('/')}/peer/__preflight_probe__?self_id=__preflight_probe_caller__"
        )
        urllib.request.urlopen(req, timeout=timeout)
        return True, "reachable (unexpected 200 on a probe id - not an error)"
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return True, "reachable (404 on unregistered probe id, as expected)"
        return False, f"reachable but returned unexpected HTTP {e.code}"
    except Exception as exc:  # noqa: BLE001
        return False, f"unreachable: {type(exc).__name__}: {exc}"


def main() -> int:
    args = build_arg_parser().parse_args()

    if (args.prev_name is None) != (args.pp_rank == 0):
        print("error: --prev-name must be omitted iff --pp-rank is 0", file=sys.stderr)
        return 2
    if (args.next_name is None) != (args.pp_rank == args.pp_world_size - 1):
        print("error: --next-name must be omitted iff --pp-rank is the last stage", file=sys.stderr)
        return 2

    if args.transport in ("udp", "quic"):
        if not args.signaling_url:
            print(f"error: --signaling-url is required for --transport {args.transport}", file=sys.stderr)
            return 2
        print(f"[preflight] checking signaling server at {args.signaling_url} ...")
        ok, detail = check_signaling_server(args.signaling_url, timeout=10.0)
        print(f"[preflight] signaling server: {'OK' if ok else 'FAIL'} - {detail}")
        if not ok:
            print(
                "[preflight] ABORT: signaling server unreachable. Verify it is "
                "running and that any tunnel (e.g. zrok) is still up before "
                "retrying. See README_RUN_GPTOSS_CLUSTER.md's Troubleshooting "
                "table ('Transport timeout').",
                file=sys.stderr,
            )
            return 1

    from vllm.transport.pipeline_bootstrap import establish_pp_transports

    local_ranks = [int(x) for x in args.local_ranks.split(",") if x != ""]
    failures: list[str] = []

    for local_rank in local_ranks:
        print(f"[preflight] local_rank={local_rank}: connecting "
              f"prev={args.prev_name} next={args.next_name} ...")
        t0 = time.monotonic()
        try:
            transport_prev, transport_next = establish_pp_transports(
                pp_rank=args.pp_rank,
                pp_world_size=args.pp_world_size,
                local_rank=local_rank,
                self_name=args.self_name,
                prev_name=args.prev_name,
                next_name=args.next_name,
                backend=args.transport,
                signaling_url=args.signaling_url,
                udp_port_base=args.udp_port_base,
                tcp_port_base=args.tcp_port_base,
                tcp_connect_host_prev=args.tcp_connect_host_prev,
                tcp_connect_host_next=args.tcp_connect_host_next,
                connect_timeout=args.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - t0
            print(f"[preflight] local_rank={local_rank}: FAIL after {elapsed:.1f}s - "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            failures.append(f"local_rank={local_rank}: {type(exc).__name__}: {exc}")
            continue

        elapsed = time.monotonic() - t0
        print(f"[preflight] local_rank={local_rank}: OK in {elapsed:.1f}s "
              f"(prev={'connected' if transport_prev else 'n/a'}, "
              f"next={'connected' if transport_next else 'n/a'})")
        if transport_prev is not None:
            transport_prev.close()
        if transport_next is not None:
            transport_next.close()

    if failures:
        print("\n[preflight] FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        print(
            "\nThis means the OTHER side of at least one link is not up yet, "
            "or not reachable. Confirm the neighbor machine(s) are running "
            "this same preflight check (or launch_pp_stage.py) concurrently - "
            "a hole punch needs both sides attempting at roughly the same "
            "time. See README_RUN_GPTOSS_CLUSTER.md's starting-order section.",
            file=sys.stderr,
        )
        return 1

    print("\nPREFLIGHT OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
