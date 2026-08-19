"""Pipeline phase / Test 5: three-stage fake pipeline, Stage0 -> Stage1 ->
Stage2. Stage1 holds two independent point-to-point connections (one to
each neighbor - Transport is point-to-point only, see
vllm/transport/README.md's TransportProcessGroup section) and forwards the
activation it receives from Stage0 to Stage2 unchanged. No computation
anywhere - only transport, through the real GroupCoordinator.send()/.recv()
interception point at every hop.

Run:
    python3 test15_pipeline_three_stage.py --transport tcp
    python3 test15_pipeline_three_stage.py --transport udp
"""
from __future__ import annotations

import argparse
import sys
import time

import torch
from _common import MP_CTX, SignalingServer, free_port, transport_config_pair
from _pipeline_shim import make_stage_coordinator

from vllm.transport import TransportConfig, get_transport

N_ELEMENTS = 256 * 1024 // 2  # 256 KB, float16
SEED = 42


def _make_activation() -> torch.Tensor:
    g = torch.Generator().manual_seed(SEED)
    return torch.randn(N_ELEMENTS, generator=g, dtype=torch.float16)


def _stage0(result_queue, backend: str, config: TransportConfig) -> None:
    transport = get_transport(backend)
    transport.connect(config)
    stage = make_stage_coordinator(transport)

    activation = _make_activation()
    t0 = time.monotonic()
    stage.send(activation)
    transport.close()
    result_queue.put({"self_id": config.self_id, "role": "stage0", "send_s": time.monotonic() - t0})


def _stage1(result_queue, backend: str, cfg_prev: TransportConfig, cfg_next: TransportConfig) -> None:
    transport_prev = get_transport(backend)
    transport_prev.connect(cfg_prev)
    stage_prev = make_stage_coordinator(transport_prev)

    transport_next = get_transport(backend)
    transport_next.connect(cfg_next)
    stage_next = make_stage_coordinator(transport_next)

    t0 = time.monotonic()
    activation = stage_prev.recv(torch.Size([N_ELEMENTS]), torch.float16, src=None)
    stage_next.send(activation)  # forward unchanged - no computation
    elapsed = time.monotonic() - t0

    transport_prev.close()
    transport_next.close()
    result_queue.put({"self_id": "Stage1", "role": "stage1", "hop_s": elapsed})


def _stage2(result_queue, backend: str, config: TransportConfig) -> None:
    transport = get_transport(backend)
    transport.connect(config)
    stage = make_stage_coordinator(transport)

    t0 = time.monotonic()
    received = stage.recv(torch.Size([N_ELEMENTS]), torch.float16, src=None)
    elapsed = time.monotonic() - t0
    transport.close()

    expected = _make_activation()
    result_queue.put({
        "self_id": config.self_id,
        "role": "stage2",
        "recv_s": elapsed,
        "equal": torch.equal(received, expected),
        "shape_ok": tuple(received.shape) == tuple(expected.shape),
    })


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
        port01 = free_port()
        port12 = free_port() if args.transport == "tcp" else port01 + 10
        cfg0, cfg1a = transport_config_pair(args.transport, "Stage0", "Stage1a", signaling_url, port01)
        cfg1b, cfg2 = transport_config_pair(args.transport, "Stage1b", "Stage2", signaling_url, port12)

        result_queue = MP_CTX.Queue()
        p1 = MP_CTX.Process(target=_stage1, args=(result_queue, args.transport, cfg1a, cfg1b))
        p2 = MP_CTX.Process(target=_stage2, args=(result_queue, args.transport, cfg2))
        p0 = MP_CTX.Process(target=_stage0, args=(result_queue, args.transport, cfg0))
        p1.start()
        p2.start()
        p0.start()

        results = [result_queue.get(timeout=60) for _ in range(3)]
        p0.join(timeout=15)
        p1.join(timeout=15)
        p2.join(timeout=15)
    finally:
        if signaling is not None:
            signaling.stop()

    stage2 = next(r for r in results if r["role"] == "stage2")
    stage1 = next(r for r in results if r["role"] == "stage1")
    stage0 = next(r for r in results if r["role"] == "stage0")

    print(f"=== Pipeline Test 5: three-stage pipeline Stage0->Stage1->Stage2 ({args.transport}) ===")
    print(f"  Stage0 send:     {stage0['send_s'] * 1000:.2f} ms")
    print(f"  Stage1 hop:      {stage1['hop_s'] * 1000:.2f} ms")
    print(f"  Stage2 recv:     {stage2['recv_s'] * 1000:.2f} ms")
    print(f"  shape_ok:        {stage2['shape_ok']}")
    print(f"  equal (Stage0's original activation == what Stage2 received): {stage2['equal']}")

    ok = stage2["shape_ok"] and stage2["equal"]
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
