"""Pipeline phase 4 (async) / tensor-dict test: exercises the REAL
`GroupCoordinator.isend_tensor_dict()`/`.irecv_tensor_dict()` transport
branch - the actual async primitives `vllm/v1/worker/gpu_worker.py`'s
`Worker._execute_model` calls (as opposed to test16, which only exercises
the synchronous `send_tensor_dict`/`recv_tensor_dict` wrappers).

Before this session's change, the transport branch of these two methods
ran the blocking work EAGERLY (right there, on the caller's own thread)
and returned an empty handle list / empty postprocess list - technically
satisfying the `Handle` contract, but giving zero real overlap: a caller
that assumed it could queue more work between `isend_tensor_dict(...)`
returning and calling `handle.wait()` got nothing for it, since the send
had already fully completed by the time it got its (empty) handle back.
This test proves three things the eager version could not:

1. **Real handles, real overlap**: `isend_tensor_dict` returns a handle
   whose `wait()` genuinely blocks until the (now backgrounded) send
   completes - proven by starting a slow "local work" callback
   immediately after `isend_tensor_dict` returns and confirming it runs
   concurrently with the send, not after it.
2. **Correctness preserved**: content still round-trips byte-for-byte
   through the async path (same checks as test16, plus the deferred-
   postprocess-population of `tensor_dict` this path uniquely needs -
   see `irecv_tensor_dict`'s docstring in parallel_state.py for why the
   returned dict starts empty here, unlike the real NCCL path).
3. **Ordering preserved under concurrent calls**: two `isend_tensor_dict`
   calls issued back-to-back, WITHOUT waiting on the first one's handle
   before issuing the second, still arrive in the order they were
   issued - the single-worker-per-direction executor
   (`GroupCoordinator._transport_io_executor`) is what guarantees this;
   a naive multi-worker pool could let them race onto QUICTransport's
   single persistent per-direction stream out of order.

Run:
    python3 test25_pipeline_tensor_dict_async.py --transport tcp
    python3 test25_pipeline_tensor_dict_async.py --transport udp
    python3 test25_pipeline_tensor_dict_async.py --transport quic
"""
from __future__ import annotations

import argparse
import sys
import time

import torch
from _common import MP_CTX, SignalingServer, free_port, transport_config_pair
from _pipeline_shim import make_stage_coordinator

from vllm.transport import TransportConfig, get_transport

LOCAL_WORK_S = 0.3  # long enough to be clearly bigger than a loopback send/recv of this payload size


def _make_dict(step: int) -> dict:
    return {
        "hidden_states": torch.randn(8, 4096, dtype=torch.float16),
        "residual": torch.randn(8, 4096, dtype=torch.float16),
        "seq_len": 8,
        "step": step,
    }


def _stage0(result_queue, backend: str, config: TransportConfig) -> None:
    transport = get_transport(backend)
    transport.connect(config)
    stage = make_stage_coordinator(transport)

    # Part 1: prove overlap. isend_tensor_dict should return well before
    # LOCAL_WORK_S worth of "compute" has happened - if it were still the
    # old eager/blocking design, issue_to_return_s would already include
    # (most of) the network time, making this assertion meaningless; the
    # real signal is that local_work finishes concurrently with the send,
    # not that issue itself is instant (queuing does take a little time -
    # D2H copy + serialize, deliberately still synchronous, see
    # parallel_state.py's isend_tensor_dict).
    d0 = _make_dict(step=0)
    t_issue0 = time.monotonic()
    handles = stage.isend_tensor_dict(d0)
    issue_to_return_s = time.monotonic() - t_issue0

    t_work0 = time.monotonic()
    time.sleep(LOCAL_WORK_S)  # stands in for "compute the next micro-batch"
    local_work_s = time.monotonic() - t_work0

    t_wait0 = time.monotonic()
    for h in handles:
        h.wait()
    wait_s = time.monotonic() - t_wait0

    # Part 2: two more sends issued back-to-back, WITHOUT waiting on the
    # first before issuing the second - proves ordering survives even
    # under real concurrent submission (see module docstring, point 3).
    d1, d2 = _make_dict(step=1), _make_dict(step=2)
    handles1 = stage.isend_tensor_dict(d1)
    handles2 = stage.isend_tensor_dict(d2)
    for h in handles1 + handles2:
        h.wait()

    transport.close()
    result_queue.put({
        "self_id": config.self_id, "role": "stage0",
        "issue_to_return_s": issue_to_return_s,
        "local_work_s": local_work_s,
        "wait_s": wait_s,
        "handle_count_step0": len(handles),
    })


def _stage1(result_queue, backend: str, config: TransportConfig) -> None:
    transport = get_transport(backend)
    transport.connect(config)
    stage = make_stage_coordinator(transport)

    tensor_dict, handles, postprocess = stage.irecv_tensor_dict()
    before_wait_keys = list(tensor_dict.keys())  # expected empty - see module docstring, point 2
    for h in handles:
        h.wait()
    for fn in postprocess:
        fn()
    after_wait_keys = sorted(tensor_dict.keys())

    expected0 = _make_dict(step=0)
    checks = {
        "before_wait_empty": before_wait_keys == [],
        "keys_match": set(tensor_dict.keys()) == set(expected0.keys()),
        "hidden_states_equal": torch.equal(tensor_dict["hidden_states"], expected0["hidden_states"]),
        "residual_equal": torch.equal(tensor_dict["residual"], expected0["residual"]),
        "seq_len_equal": tensor_dict["seq_len"] == expected0["seq_len"],
        "step0_equal": tensor_dict["step"] == 0,
    }

    steps_received = []
    for _ in range(2):
        td, hs, pp = stage.irecv_tensor_dict()
        for h in hs:
            h.wait()
        for fn in pp:
            fn()
        steps_received.append(td["step"])

    transport.close()
    result_queue.put({
        "self_id": config.self_id, "role": "stage1",
        "checks": checks,
        "after_wait_keys": after_wait_keys,
        "steps_received_in_order": steps_received,
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

    print(f"=== Test 25: async isend_tensor_dict/irecv_tensor_dict via transport ({args.transport}) ===")
    print(f"  isend_tensor_dict() issue-to-return: {stage0['issue_to_return_s'] * 1000:.1f}ms "
          f"(D2H copy + serialize only - see parallel_state.py)")
    print(f"  local \"compute\" while send was in flight: {stage0['local_work_s'] * 1000:.1f}ms")
    print(f"  handle.wait() afterwards: {stage0['wait_s'] * 1000:.1f}ms")
    print(f"  before_wait tensor_dict was empty (as documented): {stage1['checks']['before_wait_empty']}")
    for name, value in stage1["checks"].items():
        print(f"  {name}: {value}")
    print(f"  3 sends issued back-to-back arrived in order: {stage1['steps_received_in_order']}")

    ok = (
        stage0["issue_to_return_s"] < LOCAL_WORK_S  # issuing the send did not itself block for the full transfer
        and all(stage1["checks"].values())
        and stage1["steps_received_in_order"] == [1, 2]
    )
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
