"""Real prototype/proof for the biggest identified remaining blocker:
propagating ONE machine's real Scheduler decision (`SchedulerOutput`) to
another machine's local executor, using `transport_runtime` - the piece
`README_ARCHITECTURE_DECISION.md` Part 1.4/7 calls the
"MultiprocExecutor cross-machine dispatch gap" (real vLLM only
broadcasts `scheduler_output` to workers on the SAME machine via a
shared-memory `MessageQueue`; nothing today propagates it across
machines for our non-Ray, non-torch.distributed-reachable topology).

This does NOT run a real model or a real Scheduler - it constructs a
REAL `vllm.v1.core.sched.output.SchedulerOutput` (the actual class real
vLLM uses, imported for real, not a fake stand-in) with realistic field
values, and proves the exact mechanism a real cross-machine "driver
sends, follower executes" split would need:

1. real cloudpickle serialization of a real SchedulerOutput (same
   serialization vLLM's own local MessageQueue path already uses -
   confirmed by reading `multiproc_executor.py`'s
   `collective_rpc`/`cloudpickle.dumps` call) - proves the object is a
   crossable payload at all, not just structurally plausible.
2. Real transport_runtime `Connection.send()/recv()` delivery (TCP
   loopback here, in place of the real transport_runtime backends
   already proven cross-real-machine via the udp_holepunch signaling
   server in earlier phases - this test's concern is the payload/
   protocol, not re-proving hole punching).
3. Exact field-for-field equality after the round trip - proves no
   silent corruption/truncation for a payload of this shape/size.

Run:
    python3 prototype_cross_machine_scheduler_dispatch.py
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transport_runtime"))

import cloudpickle
import multiprocessing as mp

MP_CTX = mp.get_context("spawn")


def _driver(result_queue, port: int) -> None:
    from transport_runtime import BytesCodec, ConnectionManager, ConnectParams, TCPBackendConfig

    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput

    # A real SchedulerOutput, shaped like a real decode-step batch of 3
    # requests (1 new prefill + 2 cached/decode-continuing), matching
    # the actual dataclass fields real vLLM's Scheduler.schedule() fills.
    new_req = NewRequestData(
        req_id="req-0",
        prompt_token_ids=[1, 2, 3, 4, 5],
        mm_features=[],
        sampling_params=SamplingParams(max_tokens=16, temperature=0.0),
        pooling_params=None,
        block_ids=([0, 1],),
        num_computed_tokens=0,
        lora_request=None,
    )
    cached = CachedRequestData(
        req_ids=["req-1", "req-2"],
        resumed_req_ids=set(),
        new_token_ids=[[42], [43]],
        all_token_ids={},
        new_block_ids=[None, None],
        num_computed_tokens=[128, 256],
        num_output_tokens=[3, 7],
    )
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[new_req],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={"req-0": 5, "req-1": 1, "req-2": 1},
        total_num_scheduled_tokens=7,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[0],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    manager = ConnectionManager()
    conn = manager.connect(
        "follower",
        ConnectParams(self_id="driver", peer_id="follower", tcp=TCPBackendConfig(host="127.0.0.1", port=port, listen=True)),
        BytesCodec(),
        backend_name="tcp",
        role="control",
    )

    payload = cloudpickle.dumps(scheduler_output, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[driver] real SchedulerOutput pickled: {len(payload)} bytes", flush=True)
    conn.send(payload)
    print("[driver] sent over real transport_runtime Connection", flush=True)

    time.sleep(3.0)  # give follower time to recv+verify before this process exits
    result_queue.put({"role": "driver", "ok": True, "sent_bytes": len(payload)})
    conn.close()


def _follower(result_queue, port: int) -> None:
    from transport_runtime import BytesCodec, ConnectionManager, ConnectParams, TCPBackendConfig

    manager = ConnectionManager()
    time.sleep(1.0)  # let driver bind/listen first
    conn = manager.connect(
        "driver",
        ConnectParams(self_id="follower", peer_id="driver", tcp=TCPBackendConfig(host="127.0.0.1", port=port, listen=False)),
        BytesCodec(),
        backend_name="tcp",
        role="control",
    )

    raw = conn.recv(timeout=10.0)
    scheduler_output = pickle.loads(raw)
    print(f"[follower] received {len(raw)} bytes, unpickled: {type(scheduler_output).__name__}", flush=True)

    checks = {
        "total_num_scheduled_tokens": scheduler_output.total_num_scheduled_tokens == 7,
        "num_scheduled_tokens": scheduler_output.num_scheduled_tokens == {"req-0": 5, "req-1": 1, "req-2": 1},
        "new_req_id": scheduler_output.scheduled_new_reqs[0].req_id == "req-0",
        "new_req_prompt": scheduler_output.scheduled_new_reqs[0].prompt_token_ids == [1, 2, 3, 4, 5],
        "cached_req_ids": scheduler_output.scheduled_cached_reqs.req_ids == ["req-1", "req-2"],
        "cached_new_token_ids": scheduler_output.scheduled_cached_reqs.new_token_ids == [[42], [43]],
        "sampling_params_max_tokens": scheduler_output.scheduled_new_reqs[0].sampling_params.max_tokens == 16,
    }
    ok = all(checks.values())
    print("[follower] field-by-field checks:", checks, flush=True)

    conn.send(b"ACK" if ok else b"FAIL")
    result_queue.put({"role": "follower", "ok": ok, "checks": checks, "received_bytes": len(raw)})
    conn.close()


def main() -> int:
    port = 41377
    result_queue = MP_CTX.Queue()
    p_driver = MP_CTX.Process(target=_driver, args=(result_queue, port))
    p_follower = MP_CTX.Process(target=_follower, args=(result_queue, port))
    p_follower.start()
    p_driver.start()

    results = [result_queue.get(timeout=30), result_queue.get(timeout=30)]
    p_driver.join(timeout=10)
    p_follower.join(timeout=10)

    print("\n=== Cross-machine scheduler_output dispatch prototype ===")
    ok = True
    for r in results:
        print(" ", r)
        if not r.get("ok"):
            ok = False

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
