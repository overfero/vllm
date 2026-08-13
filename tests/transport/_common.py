"""Shared helpers for the communication-only transport tests.

No CUDA, no model, no torch.distributed/NCCL - these tests exercise
vllm.transport directly (TCPTransport / UDPTransport), the same way any
future non-torch-distributed worker code would.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # make `vllm` importable from a source checkout

from vllm.transport import TransportConfig  # noqa: E402

EXISTING_TRANSPORT_DIR = Path(
    os.environ.get("VLLM_UDP_TRANSPORT_DIR", str(Path(__file__).resolve().parents[2] / "udp_holepunch"))
)
MP_CTX = mp.get_context("fork")


def free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class SignalingServer:
    """A fresh instance of the existing (unmodified) signaling server, for UDP tests only."""

    def __init__(self) -> None:
        self.port = free_port()
        self.proc: subprocess.Popen | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "signaling_server:app", "--host", "127.0.0.1", "--port", str(self.port)],
            cwd=str(EXISTING_TRANSPORT_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                requests.get(f"{self.url}/peer/__probe__", timeout=1)
                return
            except requests.exceptions.RequestException:
                time.sleep(0.2)
        raise RuntimeError("signaling server did not come up in time")

    def stop(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def transport_config_pair(
    backend: str,
    id_a: str,
    id_b: str,
    signaling_url: str | None,
    base_port: int,
) -> tuple[TransportConfig, TransportConfig]:
    """One TransportConfig per side of a two-peer link. This is the *only*
    place test code branches on backend name - everything downstream (the
    worker bodies) is identical for tcp and udp."""
    if backend == "tcp":
        return (
            TransportConfig(self_id=id_a, peer_id=id_b, host="127.0.0.1", port=base_port, listen=True),
            TransportConfig(self_id=id_b, peer_id=id_a, host="127.0.0.1", port=base_port, listen=False),
        )
    if backend == "udp":
        assert signaling_url is not None
        return (
            TransportConfig(
                self_id=id_a, peer_id=id_b, signaling_url=signaling_url, udp_mode="preserve", udp_port=base_port
            ),
            TransportConfig(
                self_id=id_b, peer_id=id_a, signaling_url=signaling_url, udp_mode="preserve", udp_port=base_port + 1
            ),
        )
    raise ValueError(f"unknown backend {backend!r}")


def run_workers(worker_fn, kwargs_list: list[dict], timeout: float = 120.0) -> list[dict]:
    """Runs len(kwargs_list) worker processes concurrently. Each worker_fn
    must accept (result_queue, **kwargs) and put exactly one dict onto
    result_queue before returning. Returns the list of result dicts (order
    not guaranteed to match kwargs_list - each result dict should self-
    identify, e.g. via a "self_id" key)."""
    result_queue = MP_CTX.Queue()
    procs = [
        MP_CTX.Process(target=worker_fn, args=(result_queue,), kwargs=kwargs) for kwargs in kwargs_list
    ]
    for p in procs:
        p.start()

    results = []
    deadline = time.monotonic() + timeout
    for _ in procs:
        remaining = max(0.1, deadline - time.monotonic())
        results.append(result_queue.get(timeout=remaining))

    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)

    failed = [p for p in procs if p.exitcode not in (0, None)]
    if failed:
        raise RuntimeError(f"{len(failed)} worker process(es) exited non-zero")

    return results
