# pp_tests/ - cluster launch scripts, diagnostics, validation

## `launch/` - current canonical launch commands

`launch_machine{A,B,C,D}.sh` are the scripts that actually bring up the
current 4-machine Qwen3.5-122B-A10B-GPTQ-Int4 cluster with MTP speculative
decoding enabled - see `docs/DEPLOYMENT.md` for the full topology and what
each one does. All four read `SIGNALING_URL` from the environment (set it
before running, or export it before SSHing in). Machine D is the driver
(`launch_pp_stage.py --serve`); A/B/C are non-driver stages
(`stage_server.py`).

```bash
export SIGNALING_URL=https://your-tunnel.example.com
./launch/launch_machineA.sh      # run this one locally
# run the other three on their respective machines, same env var set
```

## `archive/` - superseded experiments, kept for reference

Earlier launch script variants from before the current 4-machine/MTP setup
(3-machine baseline, CUDA-graph-only experiments, an early 4-machine
attempt without MTP) plus two historical debugging reports
(`BLOCKER_REPORT.md`/`QWEN3.5_DEPLOYMENT_REPORT.md`, now superseded by
`docs/DEPLOYMENT.md` - see `docs/history/README.md`). Not maintained;
don't build on these, they're here for the debugging trail.

## Diagnostics and one-off validation scripts

- `real_ping_pong.py` - real cross-machine UDP round-trip latency, using
  the actual production transport path (not loopback). Run with matching
  `--role pinger`/`--role ponger` on two machines simultaneously.
- `verify_local_tp2_nccl.py` - sanity-checks the local TP=2/NCCL half of
  the topology on one machine's 2 GPUs, independent of the cross-machine
  PP transport.
- `profile_num_gpu_blocks.py` - constructs a real `EngineCore` for one
  stage without `--num-gpu-blocks-override`, to observe the naturally
  auto-profiled KV cache block count on real hardware (used to pick a safe
  uniform override across all machines - the override must not exceed the
  smallest stage's naturally-available block count).
- `test_rpc_executor_control_channel.py` - validates the wire protocol
  between `TransportExecutor._dispatch_remote` (driver side) and
  `stage_server.py`'s RPC loop (non-driver side) without needing a real
  checkpoint or GPU on both ends.
- `real_3machine_pp_test.py` - earlier real end-to-end test, predates the
  4-machine topology; kept for reference.
- `chatbot_server.py` + `chatbot_ui.html` - a minimal same-origin chat UI
  that proxies to the driver's `/v1/chat/completions` (avoids CORS setup):
  `python3 chatbot_server.py --upstream http://127.0.0.1:8080`, then open
  the forwarded port in a browser.
