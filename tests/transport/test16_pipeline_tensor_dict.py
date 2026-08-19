"""Pipeline phase 4 / tensor-dict test: exercises the REAL
`GroupCoordinator.send_tensor_dict()`/`.recv_tensor_dict()` - the actual
method `vllm/v1/worker/gpu_model_runner.py` calls for real multi-tensor
pipeline-parallel activation passing (see vllm/transport/README.md, "Phase
4"), as opposed to test11-15's single-tensor `.send()`/`.recv()`.

A dict mixing CPU tensors (standing in for e.g. "hidden_states",
"residual") and a plain non-tensor value (standing in for e.g. a scalar
metadata field) round-trips through TransportProcessGroup underneath.

Run:
    python3 test16_pipeline_tensor_dict.py --transport tcp
    python3 test16_pipeline_tensor_dict.py --transport udp
"""
from __future__ import annotations

import argparse
import sys

import torch
from _common import MP_CTX, SignalingServer, free_port, transport_config_pair
from _pipeline_shim import make_stage_coordinator

from vllm.transport import TransportConfig, get_transport


def _make_dict() -> dict:
    return {
        "hidden_states": torch.randn(8, 4096, dtype=torch.float16),
        "residual": torch.randn(8, 4096, dtype=torch.float16),
        "seq_len": 8,
        "request_id": "req-42",
    }


def _stage0(result_queue, backend: str, config: TransportConfig) -> None:
    transport = get_transport(backend)
    transport.connect(config)
    stage = make_stage_coordinator(transport)

    d = _make_dict()
    ret = stage.send_tensor_dict(d)
    transport.close()
    result_queue.put({"self_id": config.self_id, "role": "stage0", "send_return_is_none": ret is None})


def _stage1(result_queue, backend: str, config: TransportConfig) -> None:
    transport = get_transport(backend)
    transport.connect(config)
    stage = make_stage_coordinator(transport)

    received = stage.recv_tensor_dict()
    transport.close()

    expected = _make_dict()
    checks = {
        "keys_match": set(received.keys()) == set(expected.keys()),
        "hidden_states_equal": torch.equal(received["hidden_states"], expected["hidden_states"]),
        "residual_equal": torch.equal(received["residual"], expected["residual"]),
        "seq_len_equal": received["seq_len"] == expected["seq_len"],
        "request_id_equal": received["request_id"] == expected["request_id"],
        "hidden_states_dtype_ok": received["hidden_states"].dtype == torch.float16,
    }
    result_queue.put({"self_id": config.self_id, "role": "stage1", "checks": checks})


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
    stage1 = next(r for r in results if r["role"] == "stage1")

    print(f"=== Pipeline Test (tensor_dict): send_tensor_dict/recv_tensor_dict ({args.transport}) ===")
    print(f"  send_tensor_dict() returned None (matches real API contract): {stage0['send_return_is_none']}")
    ok = stage0["send_return_is_none"]
    for name, value in stage1["checks"].items():
        print(f"  {name}: {value}")
        ok = ok and value

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
