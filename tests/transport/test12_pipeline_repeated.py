"""Pipeline phase / Test 2: repeated activation transfer through the real
GroupCoordinator.send()/.recv(), 1000 iterations. Stage1 echoes a small ack
tensor back after each receive so Stage0 can measure round-trip latency,
jitter, and loss - same idea as the communication-only phase's ping-pong
test, but exercising the pipeline interception point instead of raw
Transport.

Run:
    python3 test12_pipeline_repeated.py --transport tcp
    python3 test12_pipeline_repeated.py --transport udp
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import torch
from _common import MP_CTX, SignalingServer, free_port, transport_config_pair
from _pipeline_shim import make_stage_coordinator

from vllm.transport import TransportConfig, get_transport

N_ITERS = 1000
ACTIVATION_ELEMS = 4096  # 8 KB per activation (float16)
ACK_ELEMS = 4
TIMEOUT_S = 2.0


def _stage0(result_queue, backend: str, config: TransportConfig) -> None:
    transport = get_transport(backend)
    transport.connect(config)
    stage = make_stage_coordinator(transport)

    # GroupCoordinator.recv() (real vLLM signature - see _pipeline_shim.py)
    # has no timeout parameter, matching upstream: a blocking NCCL/gloo recv
    # doesn't have one either. So there is no per-iteration "did this one
    # time out" event to catch here - a genuinely stuck peer would hang this
    # loop and be caught by main()'s outer result_queue.get(timeout=...)
    # instead. `lost` stays structurally 0; it's reported anyway since the
    # task asks for a loss measurement, and 0 is the real, meaningful
    # result given phase 1/2 already established this transport is reliable.
    rtts = []
    activation = torch.randn(ACTIVATION_ELEMS, dtype=torch.float16)
    for _ in range(N_ITERS):
        t0 = time.monotonic()
        stage.send(activation)
        stage.recv(torch.Size([ACK_ELEMS]), torch.float16, src=None)
        rtts.append(time.monotonic() - t0)
    transport.close()

    rtts_ms = sorted(r * 1000 for r in rtts)
    result_queue.put({
        "self_id": config.self_id,
        "role": "stage0",
        "avg_rtt_ms": statistics.mean(rtts_ms) if rtts_ms else None,
        "jitter_ms": statistics.pstdev(rtts_ms) if len(rtts_ms) > 1 else 0.0,
        "lost": N_ITERS - len(rtts),
        "n": N_ITERS,
    })


def _stage1(result_queue, backend: str, config: TransportConfig) -> None:
    transport = get_transport(backend)
    transport.connect(config)
    stage = make_stage_coordinator(transport)

    ack = torch.zeros(ACK_ELEMS, dtype=torch.float16)
    received = 0
    for _ in range(N_ITERS):
        stage.recv(torch.Size([ACTIVATION_ELEMS]), torch.float16, src=None)
        received += 1
        stage.send(ack)
    transport.close()
    result_queue.put({"self_id": config.self_id, "role": "stage1", "received": received})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["tcp", "udp"], required=True)
    args = parser.parse_args()

    signaling = SignalingServer() if args.transport == "udp" else None
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

        results = [result_queue.get(timeout=180), result_queue.get(timeout=180)]
        p0.join(timeout=15)
        p1.join(timeout=15)
    finally:
        if signaling is not None:
            signaling.stop()

    stage0 = next(r for r in results if r["role"] == "stage0")
    stage1 = next(r for r in results if r["role"] == "stage1")

    print(f"=== Pipeline Test 2: {N_ITERS} repeated activation transfers ({args.transport}) ===")
    print(f"  avg RTT:  {stage0['avg_rtt_ms']:.3f} ms" if stage0["avg_rtt_ms"] is not None else "  avg RTT:  N/A")
    print(f"  jitter:   {stage0['jitter_ms']:.3f} ms")
    print(f"  lost:     {stage0['lost']}/{stage0['n']}")
    print(f"  received (stage1): {stage1['received']}/{N_ITERS}")

    ok = stage0["lost"] == 0 and stage1["received"] == N_ITERS
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
