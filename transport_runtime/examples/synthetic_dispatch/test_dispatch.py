"""Resolves an open question flagged in architecture review (see
`vllm/README_ARCHITECTURE_DECISION.md`, non-blocking improvement #1 in the
final review round): does "process/worker bootstrap" belong in the same
adapter as "scheduling-step dispatch", or are they different concerns
that should be grouped differently?

This test builds a fake framework's bootstrap-once-then-dispatch-N-times
loop using nothing but `ConnectionManager.connect()` (called once) and
ordinary `Connection.send()`/`.recv()` (called once per step). No new
class or abstraction was added to `transport_runtime` to make this work -
which is itself the answer:

- Bootstrap (establishing Connections) is a genuinely distinct, one-time,
  cold-path concern - nothing here reuses it after the first call.
- Scheduling-step dispatch is NOT a distinct concern from Communication -
  it is exactly `Connection.send()`/`.recv()` of a small JSON message on
  a control-plane Connection, called repeatedly, the same primitive used
  for tensor payloads on a data-plane Connection. Grouping it with
  "Lifecycle" (as the first-draft Part 5 diagram did) was grouping by
  concept-name rather than by operational profile (hot path vs. cold
  path) - see README_ARCHITECTURE_DECISION.md's updated Part 5 for the
  corrected diagram this test justifies.

Two connections are used deliberately (control: JSONCodec, data:
TensorCodec) to mirror the control-plane/data-plane split from Part 5,
proving that split composes cleanly with a per-step dispatch loop.
"""
from __future__ import annotations

import socket
import threading

import torch

from transport_runtime import ConnectionManager, ConnectParams, JSONCodec, TCPBackendConfig, TensorCodec

_N_STEPS = 5


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_bootstrap_once_then_dispatch_n_times() -> None:
    port_control = _free_tcp_port()
    port_data = _free_tcp_port()
    errors: list[Exception] = []
    results: dict[str, object] = {}

    def scheduler_side() -> None:
        try:
            manager = ConnectionManager()
            # --- bootstrap: cold path, exactly once ---
            control = manager.connect(
                "worker",
                ConnectParams(self_id="sched-ctrl", peer_id="worker-ctrl", tcp=TCPBackendConfig(host="127.0.0.1", port=port_control, listen=True)),
                JSONCodec(),
                backend_name="tcp",
                role="control",
            )
            data = manager.connect(
                "worker",
                ConnectParams(self_id="sched-data", peer_id="worker-data", tcp=TCPBackendConfig(host="127.0.0.1", port=port_data, listen=True)),
                TensorCodec(),
                backend_name="tcp",
                role="data",
            )
            # --- dispatch: hot path, N times, reusing the same Connections ---
            acks = []
            for step in range(_N_STEPS):
                control.send({"step": step, "op": "forward"})  # opaque to the runtime - just bytes
                data.send(torch.full((2, 2), float(step)))
                acks.append(control.recv(timeout=5))
            results["acks"] = acks
            manager.close_all()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def worker_side() -> None:
        try:
            manager = ConnectionManager()
            control = manager.connect(
                "sched",
                ConnectParams(self_id="worker-ctrl", peer_id="sched-ctrl", tcp=TCPBackendConfig(host="127.0.0.1", port=port_control, listen=False)),
                JSONCodec(),
                backend_name="tcp",
                role="control",
            )
            data = manager.connect(
                "sched",
                ConnectParams(self_id="worker-data", peer_id="sched-data", tcp=TCPBackendConfig(host="127.0.0.1", port=port_data, listen=False)),
                TensorCodec(),
                backend_name="tcp",
                role="data",
            )
            received = []
            for _ in range(_N_STEPS):
                msg = control.recv(timeout=5)
                tensor = data.recv(timeout=5)
                received.append(tensor)
                control.send({"step": msg["step"], "status": "done"})
            results["received"] = received
            manager.close_all()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    t_worker = threading.Thread(target=worker_side, daemon=True)
    t_sched = threading.Thread(target=scheduler_side, daemon=True)
    t_worker.start()
    t_sched.start()
    t_sched.join(timeout=20)
    t_worker.join(timeout=20)

    if errors:
        raise errors[0]

    acks = results["acks"]
    received = results["received"]
    assert len(acks) == _N_STEPS
    assert [a["status"] for a in acks] == ["done"] * _N_STEPS
    assert [a["step"] for a in acks] == list(range(_N_STEPS))
    assert len(received) == _N_STEPS
    for step, tensor in enumerate(received):
        assert torch.equal(tensor, torch.full((2, 2), float(step)))
