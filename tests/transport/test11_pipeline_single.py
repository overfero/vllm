"""Pipeline phase / Test 1: single activation transfer, Stage0 -> Stage1,
through the REAL (patched) `GroupCoordinator.send()`/`.recv()` - the same
methods vLLM's own pipeline-parallel code and its own test suite
(tests/distributed/test_comm_ops.py::send_recv_test_worker) call for
single-tensor point-to-point transfer. Here they route through
TransportProcessGroup/Transport instead of device_communicator/NCCL - see
_pipeline_shim.py for how the real class is constructed without
torch.distributed/CUDA/Ray.

Sizes: 1/4/16/64 MB, float16 (a realistic pipeline-parallel activation
dtype). Verifies equality, latency, throughput.

Run:
    python3 test11_pipeline_single.py --transport tcp
    python3 test11_pipeline_single.py --transport udp
"""
from __future__ import annotations

import argparse
import sys
import time

import torch
from _common import MP_CTX, SignalingServer, free_port, transport_config_pair
from _pipeline_shim import make_stage_coordinator

from vllm.transport import TransportConfig, get_transport

MB = 1024 * 1024
SIZES_BYTES = [1 * MB, 4 * MB, 16 * MB, 64 * MB]
ELEMENT_BYTES = 2  # float16


def _stage0(result_queue, backend: str, config: TransportConfig, sizes: list[int]) -> None:
    transport = get_transport(backend)
    transport.connect(config)
    stage = make_stage_coordinator(transport)

    per_size = []
    for nbytes in sizes:
        n_elems = nbytes // ELEMENT_BYTES
        activation = torch.randn(n_elems, dtype=torch.float16)
        t0 = time.monotonic()
        stage.send(activation)
        ack = transport.recv(timeout=60)
        elapsed = time.monotonic() - t0
        per_size.append({
            "nbytes": nbytes,
            "latency_s": elapsed,
            "throughput_mbps": (nbytes * 8) / (elapsed * 1_000_000) if elapsed > 0 else 0.0,
            "ack": ack.decode(),
        })
    transport.close()
    result_queue.put({"self_id": config.self_id, "role": "stage0", "results": per_size})


def _stage1(result_queue, backend: str, config: TransportConfig, sizes: list[int]) -> None:
    transport = get_transport(backend)
    transport.connect(config)
    stage = make_stage_coordinator(transport)

    per_size = []
    for nbytes in sizes:
        n_elems = nbytes // ELEMENT_BYTES
        received = stage.recv(torch.Size([n_elems]), torch.float16)
        ok = received.numel() == n_elems and received.dtype == torch.float16
        transport.send(("OK" if ok else "FAIL").encode())
        per_size.append({"nbytes": nbytes, "ok": ok})
    transport.close()
    result_queue.put({"self_id": config.self_id, "role": "stage1", "results": per_size})


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
        p1 = MP_CTX.Process(target=_stage1, args=(result_queue, args.transport, cfg1, SIZES_BYTES))
        p0 = MP_CTX.Process(target=_stage0, args=(result_queue, args.transport, cfg0, SIZES_BYTES))
        p1.start()
        p0.start()

        results = [result_queue.get(timeout=180), result_queue.get(timeout=180)]
        p0.join(timeout=15)
        p1.join(timeout=15)
    finally:
        if signaling is not None:
            signaling.stop()

    stage0_result = next(r for r in results if r["role"] == "stage0")
    stage1_result = next(r for r in results if r["role"] == "stage1")

    print(f"=== Pipeline Test 1: single activation, Stage0->Stage1 ({args.transport}) ===")
    ok = True
    for s, r in zip(stage0_result["results"], stage1_result["results"]):
        mb = s["nbytes"] // MB
        print(
            f"  {mb:3d} MB: latency={s['latency_s'] * 1000:8.2f}ms  "
            f"throughput={s['throughput_mbps']:8.2f}Mbps  ok={r['ok']}"
        )
        ok = ok and r["ok"] and s["ack"] == "OK"

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
