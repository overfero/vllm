"""Pipeline phase / Test 4: bidirectional pipeline round trip.
Stage0 -> Stage1 -> Stage0: Stage0 sends an activation, Stage1 receives it
and sends it back unchanged, Stage0 receives it and verifies byte-perfect
equality against the original. All through the real GroupCoordinator.send()
/.recv() interception point.

Run:
    python3 test14_pipeline_bidirectional.py --transport tcp
    python3 test14_pipeline_bidirectional.py --transport udp
"""
from __future__ import annotations

import argparse
import sys
import time

import torch
from _common import MP_CTX, SignalingServer, free_port, transport_config_pair
from _pipeline_shim import make_stage_coordinator

from vllm.transport import TransportConfig, get_transport

N_ELEMENTS = 1 * 1024 * 1024 // 2  # 1 MB, float16


def _stage0(result_queue, backend: str, config: TransportConfig) -> None:
    transport = get_transport(backend)
    transport.connect(config)
    stage = make_stage_coordinator(transport)

    original = torch.randn(N_ELEMENTS, dtype=torch.float16)
    t0 = time.monotonic()
    stage.send(original)
    round_tripped = stage.recv(torch.Size([N_ELEMENTS]), torch.float16, src=None)
    elapsed = time.monotonic() - t0
    transport.close()

    result_queue.put({
        "self_id": config.self_id,
        "role": "stage0",
        "elapsed_s": elapsed,
        "equal": torch.equal(original, round_tripped),
    })


def _stage1(result_queue, backend: str, config: TransportConfig) -> None:
    transport = get_transport(backend)
    transport.connect(config)
    stage = make_stage_coordinator(transport)

    activation = stage.recv(torch.Size([N_ELEMENTS]), torch.float16, src=None)
    stage.send(activation)  # send back unchanged - Stage1 -> Stage0 leg
    transport.close()
    result_queue.put({"self_id": config.self_id, "role": "stage1"})


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
        cfg0, cfg1 = transport_config_pair(args.transport, "Stage0", "Stage1", signaling_url, base_port)

        result_queue = MP_CTX.Queue()
        p1 = MP_CTX.Process(target=_stage1, args=(result_queue, args.transport, cfg1))
        p0 = MP_CTX.Process(target=_stage0, args=(result_queue, args.transport, cfg0))
        p1.start()
        p0.start()

        results = [result_queue.get(timeout=60), result_queue.get(timeout=60)]
        p0.join(timeout=15)
        p1.join(timeout=15)
    finally:
        if signaling is not None:
            signaling.stop()

    stage0 = next(r for r in results if r["role"] == "stage0")

    print(f"=== Pipeline Test 4: bidirectional round trip Stage0->Stage1->Stage0 ({args.transport}) ===")
    print(f"  round-trip time: {stage0['elapsed_s'] * 1000:.2f} ms")
    print(f"  equal:           {stage0['equal']}")

    ok = stage0["equal"]
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
