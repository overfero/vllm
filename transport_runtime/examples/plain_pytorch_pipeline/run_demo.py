"""Launches the 3-stage plain-PyTorch pipeline (`stage_runner.py`) as 3
real subprocesses on loopback TCP, then verifies stage C's output matches
a directly-computed (non-pipelined) ground truth. This is the Phase 2B
exit criterion from `vllm/README_ARCHITECTURE_DECISION.md`: a pipeline
built entirely on `transport_runtime`'s Backend/Codec/Connection layers,
with zero vLLM/SGLang concepts anywhere in the stage code.
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from stage_runner import LinearStage, _SEEDS  # noqa: E402  (script, not a package import)

_STAGE_RUNNER = Path(__file__).resolve().parent / "stage_runner.py"


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_pipeline(input_seed: int = 0, timeout: float = 20.0) -> torch.Tensor:
    port_ab = _free_tcp_port()
    port_bc = _free_tcp_port()

    with tempfile.TemporaryDirectory() as tmp:
        out_paths = {role: Path(tmp) / f"{role}.json" for role in ("A", "B", "C")}

        # B must be listening on port_bc, and A on port_ab, before C/B
        # attempt to connect - launch in reverse pipeline order (C, B, A)
        # so each listener is up before its connecting peer starts.
        procs = []
        for role in ("C", "B", "A"):
            procs.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        str(_STAGE_RUNNER),
                        "--role",
                        role,
                        "--port-ab",
                        str(port_ab),
                        "--port-bc",
                        str(port_bc),
                        "--input-seed",
                        str(input_seed),
                        "--out",
                        str(out_paths[role]),
                    ]
                )
            )

        for proc in procs:
            ret = proc.wait(timeout=timeout)
            if ret != 0:
                raise RuntimeError(f"stage process exited with code {ret}")

        with open(out_paths["C"]) as f:
            result = json.load(f)
        assert result["role"] == "C"
        return torch.tensor(result["output"])


def ground_truth(input_seed: int = 0) -> torch.Tensor:
    """The same computation, single-process, no pipeline, no
    transport_runtime at all - what stage C's output must equal for the
    pipeline to be correct, not just "didn't crash"."""
    x = torch.full((1, 4), float(input_seed))
    for role in ("A", "B", "C"):
        x = LinearStage(seed=_SEEDS[role])(x)
    return x


if __name__ == "__main__":
    got = run_pipeline()
    expected = ground_truth()
    print("pipeline output:", got)
    print("ground truth:   ", expected)
    assert torch.allclose(got, expected), "pipeline output does not match ground truth"
    print("OK: pipeline output matches non-pipelined ground truth")
