# GPT-OSS-120B across 3 real machines (TP=2, PP=3, custom UDP transport) - status

Last updated: 2026-08-03, end of the session that got this to real end-to-end
HTTP 200 completions across 3 real machines for the first time.

**tl;dr: the distributed systems part works. The model output is garbage.**
The custom UDP hole-punch transport, the 3-machine pipeline-parallel split,
and the RPC scheduling mechanism are all now proven correct (see "What's
DONE" below). What's NOT done is model correctness: generated text is
incoherent even with greedy (temperature=0) decoding, and it's been
root-caused to `hidden_states` collapsing to exactly `0.0` inside the
decoder layers - see "What's NOT done" for the exact evidence and where to
pick up debugging.

---

## Architecture

Real GPT-OSS-120B (GPTQ-quantized, 36 layers, hidden_size=2880,
vocab_size=201088), split 3 ways because no single machine's 2xTesla T4
(2x16GB) can hold the full ~61GB checkpoint:

| Machine | role | pp-rank | layers | globals |
|---|---|---|---|---|
| A (this sandbox, local) | non-driver stage | 0 | 0-11 | embed_tokens |
| B (akun1) | non-driver stage | 1 | 12-23 | none |
| C (akun6) | **driver** - runs the real Scheduler, serves the API | 2 | 24-35 | norm, lm_head |

Each machine: `--tensor-parallel-size 2` (its own 2 T4s). Real vLLM's own
`torch.distributed`/NCCL cannot cross NAT'd machines, so this project
replaces `MultiprocExecutor` scheduling + PP tensor transfer with a custom
UDP hole-punch transport (`udp_holepunch/peer.py`, pre-existing/unmodified)
wrapped as a vLLM `Transport` (`vllm/transport/`). A public zrok-tunneled
signaling server does STUN-like rendezvous; actual data flows machine-to-
machine directly once punched through.

- **Non-driver stages (A, B)**: run `scripts/stage_server.py` - builds a
  real `EngineCore` directly (loads its shard, inits KV cache, connects PP
  tensor link(s)), then loops receiving `(method, args, kwargs)` RPC calls
  from the driver and applying them to its local `MultiprocExecutor`.
- **Driver (C)**: `scripts/launch_pp_stage.py --serve` execs into the real,
  unmodified `vllm serve` CLI with
  `--distributed-executor-backend vllm.transport.rpc_executor.TransportExecutor`
  and `--worker-cls vllm.transport.pp_worker.TransportPPWorker`.
  `TransportExecutor` fans out `execute_model`/`sample_tokens`/
  `execute_dummy_batch` to A and B over a dedicated RPC control-channel
  `Transport` connection (separate socket/port from the PP tensor link).

---

## What's DONE (this session, confirmed with real hardware/real requests)

1. **Real per-machine environment**: torch/vLLM install (`VLLM_USE_PRECOMPILED=1`),
   humming-kernels + `humming_fix` SM75 patches, `transport_runtime`,
   `udp_holepunch` synced/working on all 3 machines. Automated via
   `ops/setup_machine.sh` (idempotent, re-run-safe).
2. **Real checkpoint sharding**: each machine independently downloads the
   full GPTQ checkpoint from HF
   (`hf download positron-ai/openai_gpt-oss-120b-ingest-best-gptq --local-dir /data/models/gpt-oss-120b-gptq`,
   ~61GB) and extracts its own 12-layer shard locally via
   `humming_fix/single_layer_probe/extract_stage_checkpoint.py`, then
   deletes the full checkpoint (disk quota rule - see
   `memory/feedback_kaggle_working_placement.md`). **No large file is ever
   transferred machine-to-machine** (5GB/day bandwidth rule) - see
   `ops/README.md`'s stage/topology table for exact `--extract-stage` args.
3. **UDP hole-punch transport**: confirmed working end-to-end across all 3
   real machines - PP tensor links (A<->B, B<->C) and RPC control-channel
   links (C<->A, C<->B) all establish successfully.
4. **Real weight loading**: all 3 machines load their real GPTQ shard
   without error (`_load_weights_other`'s per-expert-indexed-checkpoint gap
   - see fix #1 below).
5. **Real end-to-end request**: a `curl` to Machine C's
   `/v1/completions` now gets a real `200 OK` with real generated tokens,
   having been scheduled by C's real Scheduler, executed identically on A/B
   via RPC, with real activation tensors relayed A->B->C over the custom
   transport. **This was blocked by 6 distinct real bugs, all fixed and
   confirmed this session** (see below).
6. **Transport correctness independently verified** via debug
   instrumentation (`VLLM_TRANSPORT_DEBUG_TENSOR_STATS=1`, see below): the
   exact tensor stats (mean/std) sent by one stage match, bit-for-bit, what
   the next stage receives, at every hop, for every step of a real request.
   The transport is not the source of the remaining bug.

### Real bugs fixed this session (all confirmed via actual tracebacks on real hardware, not guessed)

1. **`vllm/model_executor/models/gpt_oss.py`** (+81 lines): `_load_weights_other`
   (the GPTQ/AWQ loading path) never handled a per-expert-indexed checkpoint
   tensor name (e.g. `layers.0.mlp.experts.3.w2_bias`) - only
   `_load_weights_mxfp4` had the expert-id-extraction + `.data[expert_id]`
   scatter logic. Added `_scatter_per_expert_indexed_param()` plus
   dimension-aware slicing fallbacks in the `.w13_bias`/`.w2_bias`
   branches. Fixes real `KeyError`/`IndexError` crashes during weight load.
2. **`vllm/transport/pipeline_bootstrap.py`** (+22 lines):
   `install_transport_pp_group()`'s synthetic `GroupCoordinator` (built via
   `object.__new__()`, bypassing `__init__`) was missing
   `device_communicator`/`mq_broadcaster` attributes real vLLM code reads.
   Set both to `None` explicitly. Fixes real `AttributeError`s.
3. **`vllm/transport/udp_transport.py`** (+48 lines), two independent fixes:
   - **NAT keepalive**: `punch_loop` only ran during initial connect, then
     was cancelled - an idle RPC control-channel (minutes of no traffic
     while another stage loads weights) let the NAT mapping silently
     expire. Added a periodic keepalive ping (every 15s, reusing the
     existing ping/pong tags) for the life of every connection.
   - **`send_multi()`**: initially misdiagnosed a symptom (one peer's
     address appeared to flap between 2 IPs, logged as repeated "NAT
     REBINDING DETECTED") as NAT flakiness and added redundant-address
     sending as a mitigation. Turned out to be a RED HERRING for the real
     bug (#6 below) but is harmless/kept as defense-in-depth.
4. **`--enforce-eager` required on ALL 3 stages**: `stage_server.py`
   already hardcoded this; `launch_pp_stage.py`'s driver launch did not,
   so the driver (C) compiled CUDA graphs and padded batches to the
   nearest capture size (e.g. 8) while A/B ran eager/unpadded (e.g. 5) -
   `RuntimeError: size of tensor a (8) must match tensor b (5)` gathering
   intermediate tensors. Now added to `launch_pp_stage.py`'s driver cmd.
5. **`--no-async-scheduling` required on ALL 3 stages**: vLLM's default
   async scheduling makes every non-last PP rank call
   `torch.distributed.broadcast(group=pp.device_group)` to pull sampled
   token ids directly - a second, separate cross-rank channel the
   synthetic transport-backed PP group has no real backing for
   (`AttributeError: 'GroupCoordinator' object has no attribute
   'device_group'`). Disabled via CLI flag (driver) and
   `EngineArgs(async_scheduling=False)` (stage_server.py).
6. **THE port-collision bug (`vllm/transport/rpc_executor.py`, real
   root cause of the "NAT flapping" symptom)**: `_connect_remote_stages()`
   bound **every** remote-stage link to the same local UDP port
   (`rpc_port`, default 40000). With 2 remote stages (A and B),
   `SO_REUSEADDR` let Machine C's driver bind 2 sockets to
   `0.0.0.0:40000` silently, and the kernel non-deterministically
   delivered inbound packets from A or B to whichever socket - so C's
   "MachineA" connection sometimes received MachineB's packets, which
   looked exactly like the peer's address flapping. Confirmed via an
   isolated, model-free diagnostic (`scripts/diag_transport.py` - a single
   A<->C link with no model loading worked perfectly, 0.04s round trip, no
   flapping at all, proving it wasn't general NAT flakiness). Fixed by
   giving each remote-stage link its own distinct local port
   (`rpc_port + i`).

---

## What's NOT done: model output is garbage

A real request through the full pipeline now succeeds (HTTP 200), but:

```
prompt: "The capital of France is", temperature=0 (greedy/deterministic)
output: " \" this \"?!\n\n ' ' \"\n\n\n\n\n\n\n\n\n\n!!"
```

Already ruled out:
- **Not sampling noise** - identical garbage at `temperature=1` and `temperature=0`.
- **Not tokenizer mismatch** - `tokenizer.json` is byte-identical (same md5)
  across all 3 stage checkpoints.
- **Not `lm_head` corruption** - `lm_head.weight` is `[201088, 2880]` bf16,
  no NaN, not all-zero, matches `config.json`'s `vocab_size`; tokenizer's
  real vocab (200019) comfortably fits inside it (normal padding).
- **Not the transport** - added debug instrumentation
  (`VLLM_TRANSPORT_DEBUG_TENSOR_STATS=1` env var, patches
  `vllm/distributed/parallel_state.py`'s `_transport_send_tensor_dict`/
  `_transport_recv_tensor_dict` to print `mean`/`std`/`isnan`/`isinf` for
  every tensor sent/received) and confirmed **bit-for-bit fidelity** at
  every hop:
  ```
  A SEND residual: mean=0.0994616 std=12.7098   ->  B RECV residual: mean=0.0994616 std=12.7098  (identical)
  B SEND residual: mean=3.65962   std=251.638   ->  C RECV residual: mean=3.65962   std=251.638   (identical)
  ```
- **The checkpoint's own weights are fine** - `input_layernorm.weight`/
  `post_attention_layernorm.weight` tensors on disk have normal
  mean/std (~1.5-2.0 mean, non-zero, non-degenerate).

**The actual finding**: `hidden_states` (the tensor named `"hidden_states"`
in the PP hand-off dict) is **exactly** `mean=0, std=0` - i.e. every single
element is precisely `0.0` - at **every** PP boundary, on **every** machine,
starting from the very first prefill step. Meanwhile `residual` (the other
half of the hand-off dict) has real, growing, non-degenerate values. Since
transport fidelity is proven, this means Machine A's own decoder-layer
computation is producing a `hidden_states=0` tensor internally, before it's
ever sent - not a transport corruption.

Traced to `vllm/model_executor/models/gpt_oss.py`:
- `GptOssModel.forward()` (~line 401-402): for non-last PP ranks, returns
  `IntermediateTensors({"hidden_states": x, "residual": residual})` where
  `x` is the running decoder-layer output - normally the fused-add-RMSNorm
  output fed to the next sublayer, and should never legitimately be all-zero.
- `GptOssDecoderLayer.forward()` (~line 274-289): uses vLLM's fused 2-arg
  `RMSNorm.__call__(hidden_states, residual)` form
  (`input_layernorm`/`post_attention_layernorm`), which is supposed to
  return `(new_normalized_hidden_states, updated_residual_accumulator)`.

**Leading hypothesis**: a bug in the fused add+RMSNorm computation
specifically on this SM75 (Tesla T4, Turing) + GPTQ + MoE combination -
possibly in vLLM's own fused-add-rmsnorm CUDA kernel dispatch for this
hardware/shape, though note the actual MoE GEMM at runtime uses vLLM's
built-in Triton `WNA16` kernels (`Falling back to Moe WNA16 kernels` in the
logs - Marlin isn't supported on SM75), **not** the `humming-kernels`
package `humming_fix/patch.py` patches - so `humming_fix` is likely NOT the
culprit for this specific bug (it fixes a different, already-working code
path). This was NOT independently confirmed before the session ended.

**Not yet done, next steps for whoever picks this up**:
1. Add per-layer (not just per-PP-boundary) debug prints inside
   `GptOssDecoderLayer.forward()` - print `x.mean()`/`x.std()` right after
   each of `input_layernorm`, `self.attn`, `post_attention_layernorm` calls,
   for layer 0 specifically, to find the EXACT operation where `x` first
   becomes all-zero.
2. Once found, check whether it's a fused-kernel dispatch issue (does
   vLLM's `RMSNorm` module pick a different `forward_cuda`/`forward_native`
   path in this environment that has a bug?) vs. a weight-application bug
   specific to this checkpoint's tensor layout.
3. A useful control: run `vllm/model_executor/layers/layernorm.py`'s
   `RMSNorm` standalone (real weight, real random input tensor of the right
   shape/dtype, single machine, no PP/transport at all) and check if IT
   alone produces zero output on this hardware - would immediately confirm
   or rule out "RMSNorm kernel itself is broken on SM75" independent of
   the whole distributed setup.

---

## How to reproduce from a fresh session

Credentials/URLs below (ports, passwords, zrok signaling URL) are
**session-specific and WILL be different on a fresh Kaggle session** -
replace with whatever the user provides. Everything else (paths, commands,
flags) should be reused as-is.

### 0. Prerequisites already in this repo (don't redo)

All 6 fixes listed under "What's DONE" above are already applied in this
`vllm/` checkout. If starting from a clean `vllm-project/vllm` clone
instead, re-apply them (search this file's fix list + the inline code
comments at each location for exact context).

### 1. Bring up each remote machine's environment + checkpoint shard

```bash
# From this sandbox (orchestrator), one command per remote machine:
ops/setup_machine.sh --port <PORT_B> --password '<PASSWORD_B>' --name akun1 \
    --extract-stage 12:24:/data/stage1-checkpoint

ops/setup_machine.sh --port <PORT_C> --password '<PASSWORD_C>' --name akun6 \
    --extract-stage 24:36:/data/stage2-checkpoint:globals
```

Or edit `ops/machines.conf` (format: `name port password`, one per line)
and run `ops/setup_all.sh` for all machines in parallel.

On the LOCAL machine (Machine A, this sandbox - not scripted the same way,
it's the orchestrator itself):
```bash
hf download positron-ai/openai_gpt-oss-120b-ingest-best-gptq --local-dir /data/models/gpt-oss-120b-gptq
cd humming_fix/single_layer_probe
python3 extract_stage_checkpoint.py --start 0 --end 12 --out /data/stage0-checkpoint --include-globals
rm -rf /data/models  # disk quota rule - never keep full checkpoint + shard simultaneously
```

### 2. Confirm the signaling server + zrok tunnel are up

```bash
# Local signaling server (if not already running):
cd udp_holepunch && python3 -m uvicorn signaling_server:app --host 0.0.0.0 --port 8765 &
# Public zrok tunnel to it (get a fresh URL each session):
zrok2 share public http://127.0.0.1:8765 --headless &
```
Note the resulting `https://<token>.share.zrok.io` URL - this is
`$SIGNALING_URL` below.

### 3. Launch all 3 stages SIMULTANEOUSLY (critical - staggered launches race on hole-punch timing)

```bash
cd /kaggle/working/vllm
SIGNALING_URL="https://<token>.share.zrok.io"   # from step 2

# Machine A (local)
(python3 scripts/stage_server.py --model /data/stage0-checkpoint --tensor-parallel-size 2 \
  --pp-rank 0 --pp-world-size 3 --self-name MachineA --next-name MachineB --driver-name MachineC \
  --transport udp --signaling-url "$SIGNALING_URL" \
  --quantization gptq --dtype float16 --max-model-len 128 --num-gpu-blocks-override 512 \
  --transport-connect-timeout 900 > /tmp/machineA.log 2>&1 &)

# Machine B (akun1)
(sshpass -p '<PASSWORD_B>' ssh -o StrictHostKeyChecking=no -p <PORT_B> root@127.0.0.1 \
  "cd /kaggle/working/vllm; python3 scripts/stage_server.py \
  --model /data/stage1-checkpoint --tensor-parallel-size 2 \
  --pp-rank 1 --pp-world-size 3 --self-name MachineB --prev-name MachineA --next-name MachineC \
  --driver-name MachineC --transport udp --signaling-url $SIGNALING_URL \
  --quantization gptq --dtype float16 --max-model-len 128 --num-gpu-blocks-override 512 \
  --transport-connect-timeout 900" > /tmp/machineB.log 2>&1 &)

# Machine C (akun6, the driver - serves the API)
(sshpass -p '<PASSWORD_C>' ssh -o StrictHostKeyChecking=no -p <PORT_C> root@127.0.0.1 \
  "cd /kaggle/working/vllm; python3 scripts/launch_pp_stage.py --serve \
  --model /data/stage2-checkpoint --tensor-parallel-size 2 \
  --pp-rank 2 --pp-world-size 3 --self-name MachineC --prev-name MachineB \
  --remote-stage-names MachineA,MachineB --remote-stage-hosts 127.0.0.1,127.0.0.1 \
  --transport udp --signaling-url $SIGNALING_URL \
  --quantization gptq --dtype float16 --max-model-len 128 --num-gpu-blocks-override 512 \
  --transport-connect-timeout 900 --port 8080" > /tmp/machineC.log 2>&1 &)
```

`--num-gpu-blocks-override 512` MUST be identical on all 3 launches (see
`rpc_executor.py`'s module docstring for why). `--enforce-eager` and
`--no-async-scheduling` are already hardcoded/added on the driver's launch
path (`launch_pp_stage.py`) and `stage_server.py` - don't need to be passed
explicitly, just don't remove them if editing those files.

To get the model-computation debug prints described above, add
`VLLM_TRANSPORT_DEBUG_TENSOR_STATS=1` before each of the 3 python3 launch
commands.

### 4. Wait for readiness (~3 min for weight loading), then test

Watch for `[stage_server] EngineCore ready` + `RPC control channel to
driver 'MachineC' connected` on A/B's logs, and `API server: HTTP server
started` on C's log. Then:

```bash
sshpass -p '<PASSWORD_C>' ssh -o StrictHostKeyChecking=no -p <PORT_C> root@127.0.0.1 \
  'curl -s -m 90 http://127.0.0.1:8080/v1/completions -H "Content-Type: application/json" \
   -d "{\"model\": \"/data/stage2-checkpoint\", \"prompt\": \"The capital of France is\", \"max_tokens\": 16, \"temperature\": 0}"'
```

A `200` response with SOME generated text confirms the distributed
transport/scheduling side still works (this is now expected). Coherent
text does NOT yet happen - see "What's NOT done" above.

### 5. Cleanup between attempts

Kill ALL processes (not just the top-level launcher - `VLLM::Worker`/
`VLLM::EngineCore` subprocesses survive a lone `kill -9` on the parent) and
confirm `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`
shows 0 MiB on all GPUs on all 3 machines before relaunching, or a stale
process will hold GPU memory and the next launch will fail with a memory
error.
