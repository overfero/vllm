"""Pipeline phase / Test 3: simulated autoregressive decode loop. Stage0
repeatedly sends a single-token-sized activation and receives one back from
Stage1 (which just echoes it - no transformer, no model, no attention),
mimicking the send/receive shape of one pipeline-parallel hop per decode
step. Measures a tokens/sec *equivalent* - i.e. decode steps/sec this
communication path alone could sustain, with zero compute in between.

Run:
    python3 test13_pipeline_decoder_loop.py --transport tcp
    python3 test13_pipeline_decoder_loop.py --transport udp
"""
from __future__ import annotations

import argparse
import sys
import time

import torch
from _common import MP_CTX, SignalingServer, free_port, transport_config_pair
from _pipeline_shim import make_stage_coordinator

from vllm.transport import TransportConfig, get_transport

N_STEPS = 512
HIDDEN_SIZE = 4096  # a realistic single-token hidden-state width


def _stage0(result_queue, backend: str, config: TransportConfig) -> None:
    transport = get_transport(backend)
    transport.connect(config)
    stage = make_stage_coordinator(transport)

    activation = torch.randn(1, HIDDEN_SIZE, dtype=torch.float16)  # (batch=1, hidden) - one token
    t0 = time.monotonic()
    for _ in range(N_STEPS):
        stage.send(activation)
        activation = stage.recv(torch.Size([1, HIDDEN_SIZE]), torch.float16, src=None)
    elapsed = time.monotonic() - t0
    transport.close()

    result_queue.put({
        "self_id": config.self_id,
        "role": "stage0",
        "elapsed_s": elapsed,
        "steps_per_sec": N_STEPS / elapsed if elapsed > 0 else 0.0,
    })


def _stage1(result_queue, backend: str, config: TransportConfig) -> None:
    transport = get_transport(backend)
    transport.connect(config)
    stage = make_stage_coordinator(transport)

    for _ in range(N_STEPS):
        activation = stage.recv(torch.Size([1, HIDDEN_SIZE]), torch.float16, src=None)
        stage.send(activation)  # no computation - pure passthrough, per the "no model" constraint
    transport.close()
    result_queue.put({"self_id": config.self_id, "role": "stage1"})


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

        results = [result_queue.get(timeout=120), result_queue.get(timeout=120)]
        p0.join(timeout=15)
        p1.join(timeout=15)
    finally:
        if signaling is not None:
            signaling.stop()

    stage0 = next(r for r in results if r["role"] == "stage0")

    print(f"=== Pipeline Test 3: {N_STEPS}-step simulated decode loop ({args.transport}) ===")
    print(f"  elapsed:          {stage0['elapsed_s']:.3f} s")
    print(f"  steps/sec (tok/s equivalent): {stage0['steps_per_sec']:.1f}")

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
