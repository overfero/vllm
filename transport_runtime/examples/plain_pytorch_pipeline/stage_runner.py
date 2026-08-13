"""One stage of a 3-stage plain-PyTorch pipeline, running entirely on top
of `transport_runtime`, with no inference-framework or scheduler concepts
anywhere in this file.

This file specifically is the Phase 2B litmus test described in this
project's architecture decision document: `test_demo.py` greps this exact
file's source for a short list of framework-specific terms and fails the
suite if any appear, so a future edit that quietly reintroduces framework
coupling here gets caught automatically, not just by a one-time manual
read.

Each stage is one OS process (launched by `run_demo.py` via `subprocess`,
one loopback TCP port pair per inter-stage link) - deliberately not
threads-in-one-process, so this looks like what a real multi-machine
deployment would do, just on loopback instead of real hosts.
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn as nn

from transport_runtime import ConnectionManager, ConnectParams, TCPBackendConfig, TensorCodec

_SEEDS = {"A": 1, "B": 2, "C": 3}


class LinearStage(nn.Module):
    """A trivial one-layer "model" - standing in for whatever a real
    pipeline stage would compute. Deterministic from `seed` alone, so a
    verifier process can reconstruct the same weights independently and
    check the pipeline's output against a non-pipelined ground truth."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.weight = nn.Parameter(torch.randn(4, 4, generator=generator))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["A", "B", "C"], required=True)
    parser.add_argument("--port-ab", type=int, required=True)
    parser.add_argument("--port-bc", type=int, required=True)
    parser.add_argument("--input-seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manager = ConnectionManager()
    model = LinearStage(seed=_SEEDS[args.role])
    codec = TensorCodec()

    # Convention: on each A-B / B-C link, the lower-alphabetical side
    # listens and the higher side connects - an arbitrary but consistent
    # rule. Nothing about that convention lives in transport_runtime
    # itself - it's a decision this demo script makes on its own, exactly
    # as the runtime's Non-Goals says a topology/ordering choice like this
    # should be made by the caller, not the runtime.
    if args.role == "A":
        conn_next = manager.connect(
            "B",
            ConnectParams(self_id="A", peer_id="B", tcp=TCPBackendConfig(host="127.0.0.1", port=args.port_ab, listen=True)),
            codec,
            backend_name="tcp",
        )
        x = torch.full((1, 4), float(args.input_seed))
        y = model(x)
        conn_next.send(y)
        output = y
    elif args.role == "B":
        conn_prev = manager.connect(
            "A",
            ConnectParams(self_id="B", peer_id="A", tcp=TCPBackendConfig(host="127.0.0.1", port=args.port_ab, listen=False)),
            codec,
            backend_name="tcp",
        )
        conn_next = manager.connect(
            "C",
            ConnectParams(self_id="B", peer_id="C", tcp=TCPBackendConfig(host="127.0.0.1", port=args.port_bc, listen=True)),
            codec,
            backend_name="tcp",
        )
        x = conn_prev.recv(timeout=10)
        y = model(x)
        conn_next.send(y)
        output = y
    else:  # C, the last stage
        conn_prev = manager.connect(
            "B",
            ConnectParams(self_id="C", peer_id="B", tcp=TCPBackendConfig(host="127.0.0.1", port=args.port_bc, listen=False)),
            codec,
            backend_name="tcp",
        )
        x = conn_prev.recv(timeout=10)
        output = model(x)

    manager.close_all()
    with open(args.out, "w") as f:
        json.dump({"role": args.role, "output": output.tolist()}, f)


if __name__ == "__main__":
    main()
