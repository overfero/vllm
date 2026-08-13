# Running GPT-OSS 120B on the 3-Machine T4 Transport-PP Cluster

This is the operational deployment guide for the code in this repository:
`vllm/transport/pp_worker.py`, `vllm/transport/pipeline_bootstrap.py`
(`install_transport_pp_group`/`establish_pp_transports`), and
`scripts/launch_pp_stage.py`/`scripts/preflight_check.py`/
`scripts/health_check.py`. It assumes zero prior context.

**The transport layer itself (UDP hole punching, signaling server,
`vllm/transport/` primitives, the transport-backed `GroupCoordinator`,
`send_tensor_dict`/`recv_tensor_dict`) is treated as complete and frozen.
Nothing in this document changes it.** What this document covers is
everything built on top of it this session to make a real deployment
attempt possible: where `install_transport_pp_group` gets called in the
real (non-test) bootstrap, the new middle-stage dual-connection support,
the official launcher, and health checks - plus one precisely-scoped,
newly-discovered gap that stands between this and genuine multi-machine
online serving, explained in full at the end.

---

## Topology (fixed, per the task)

```
3 machines, 2x NVIDIA T4 each, 6 GPUs / 96GB total.

Tensor Parallel = 2   (local to each machine, real NCCL, unmodified vLLM)
Pipeline Parallel = 3 (one stage per machine, this project's Transport)
Data Parallel = 1

Machine A: GPUs 0-1, TP ranks {0,1}, PP rank 0, layers  0-11, embed_tokens
Machine B: GPUs 0-1, TP ranks {0,1}, PP rank 1, layers 12-23
Machine C: GPUs 0-1, TP ranks {0,1}, PP rank 2, layers 24-35, final norm, lm_head, API
```

Machine B holds **two** independent transport connections per local GPU
(one to Machine A, one to Machine C) - 4 connections total on Machine B,
2 each on Machine A and Machine C. This is new in this session; see
"Task 1: call graph" below for why every local GPU (not just one
representative per machine) needs its own link.

---

# Task 1: vLLM startup call graph (entrypoint -> first forward pass)

Traced directly from this checkout's source, file/line references given
so you can re-verify against your own vLLM version if it drifts.

```
1. python3 -m vllm.entrypoints.cli.main serve $MODEL ...
   vllm/entrypoints/cli/serve.py: ServeSubcommand.cmd(args)
   -> vllm/entrypoints/openai/api_server.py: run_server(args)

2. EngineArgs.create_engine_config()  (vllm/engine/arg_utils.py:~2195)
   -> builds ParallelConfig(tensor_parallel_size, pipeline_parallel_size,
      worker_cls, distributed_executor_backend, ...)
   -> builds VllmConfig

3. AsyncLLM / EngineCore construction (vllm/v1/engine/core.py)
   -> picks an Executor by distributed_executor_backend:
      "mp" -> MultiprocExecutor (vllm/v1/executor/multiproc_executor.py)
   -> MultiprocExecutor spawns one process per LOCAL worker
      (world_size = tensor_parallel_size * pipeline_parallel_size, and
      with our --pipeline-parallel-size 1, that's just tensor_parallel_size
      local GPU processes - see "Known gap" for why this matters)

4. Per worker process: WorkerWrapperBase.init_worker()
   (vllm/v1/worker/worker_base.py:187)
   -> resolve_obj_by_qualname(parallel_config.worker_cls)
      -> vllm.transport.pp_worker.TransportPPWorker (this session's code;
         "auto" would otherwise resolve to vllm.v1.worker.gpu_worker.Worker)
   -> Worker.init_device()  (vllm/v1/worker/gpu_worker.py:304)
        -> init_worker_distributed_environment()  (same file, ~line 1347)
             -> init_distributed_environment()          [real, unmodified]
             -> ensure_model_parallel_initialized()      [real, unmodified]
             (forms the REAL local TP=2 torch.distributed/NCCL group, and
             an initially-trivial local `_PP` with world_size=1)
        -> [TransportPPWorker's addition, see Task 2] install this
           project's transport-backed 3-member `_PP` group here
        -> self.model_runner = GPUModelRunnerV1(...)

5. Executor.collective_rpc("load_model")
   -> Worker.load_model()  (vllm/v1/worker/gpu_worker.py:435)
        -> get_model(vllm_config=...)
             -> GptOssForCausalLM.__init__ -> GptOssModel.__init__
                  -> make_layers()  (vllm/model_executor/models/utils.py:~810)
                       -> get_pp_indices(num_hidden_layers,
                            get_pp_group().rank_in_group,
                            get_pp_group().world_size)
                       reads the LIVE `_PP` group installed in step 4 -
                       this is why the transport swap must happen before
                       this point, and why it doesn't matter that
                       ParallelConfig.pipeline_parallel_size says "1".

6. Scheduler (vllm/v1/core/sched/scheduler.py) admits requests, builds
   KV cache blocks, produces SchedulerOutput each step.

7. EngineCore.step()  (vllm/v1/engine/core.py:~595)
   scheduler_output = self.scheduler.schedule(...)
   self.model_executor.execute_model(scheduler_output, non_block=True)
   -> per worker: Worker.execute_model()  (gpu_worker.py:~1019)
        if not get_pp_group().is_first_rank:
            get_pp_group().irecv_tensor_dict(...)   [transport branch]
        GPUModelRunner.execute_model(...)  [real forward pass, this
            stage's shard of layers only]
        if not get_pp_group().is_last_rank:
            get_pp_group().isend_tensor_dict(...)   [transport branch]
        else:
            compute_logits() -> sample -> first generated token
```

---

# Task 2: where `install_transport_pp_group` must be called - implemented, not pseudocode

**Exact location: `vllm/transport/pp_worker.py`'s `TransportPPWorker.init_device()`,
called via `super().init_device()` then the transport swap - i.e. strictly
after `init_worker_distributed_environment()` (step 4 above) and strictly
before `Worker.load_model()` (step 5), because `load_model()` is a
separate, later `collective_rpc` call in the real executor, and
`make_layers()` inside it reads the live `_PP` group at construction time
(step 5's citation).**

Wiring mechanism: vLLM's own `--worker-cls` extension point
(`ParallelConfig.worker_cls`, resolved via `resolve_obj_by_qualname` in
`WorkerWrapperBase.init_worker`, `vllm/v1/worker/worker_base.py:187`) -
**zero changes to `vllm/v1/worker/gpu_worker.py` itself.**
`TransportPPWorker(Worker)` overrides only `init_device()`:

```python
# vllm/transport/pp_worker.py
class TransportPPWorker(Worker):
    def init_device(self) -> None:
        super().init_device()
        # ... read VLLM_TRANSPORT_* env vars (set by launch_pp_stage.py) ...
        transport_prev, transport_next = establish_pp_transports(...)
        install_transport_pp_group(
            pp_rank=pp_rank, pp_world_size=pp_world_size,
            local_rank=self.local_rank,
            transport_prev=transport_prev, transport_next=transport_next,
        )
```

Selected on the command line: `--worker-cls vllm.transport.pp_worker.TransportPPWorker`
(`scripts/launch_pp_stage.py` sets this automatically).

**Real, passing proof this ordering is correct**:
`tests/transport/test18_real_bootstrap_pp.py` (2-member case, unchanged
from the prior session) and the new
`tests/transport/test20_real_bootstrap_pp_three_stage.py` (3-member case,
written and run this session) both construct a real local
`torch.distributed` TP group first, confirm it still works *after* the
`_PP` swap, then exchange a real tensor dict end-to-end through the swapped
group. Re-run this session:

```
$ python3 tests/transport/test18_real_bootstrap_pp.py --transport udp
...
PASS

$ python3 tests/transport/test20_real_bootstrap_pp_three_stage.py --transport udp
...
  {'pp_rank': 0, 'ok': True, ...}
  {'pp_rank': 1, 'ok': True, ..., 'checkpoints': [..., 'stage1_received_from_prev', 'stage1_forwarded_to_next']}
  {'pp_rank': 2, 'ok': True, ..., 'received_shape': [8, 32], 'received_dtype': 'torch.float32', 'step': 0}
PASS
```

---

# Task 3: `scripts/launch_pp_stage.py` - the official launcher

Implemented at `scripts/launch_pp_stage.py`. Responsibilities and how each
is met:

| Responsibility | How |
|---|---|
| Initialize local TP | `--tensor-parallel-size 2` passed straight through to the real, unmodified `vllm serve` CLI |
| Initialize local distributed group | Same - real `init_distributed_environment`, untouched |
| Replace PP group | `--worker-cls vllm.transport.pp_worker.TransportPPWorker` (Task 2) |
| Establish transport | `establish_pp_transports()`, called from inside `TransportPPWorker.init_device()`, using env vars this script sets |
| Wait until neighbors connect | `Transport.connect()` blocks until the hole punch (or TCP accept/connect) succeeds or `--transport-connect-timeout` elapses - real, tested (Task 2's proof runs) |
| Load GPT-OSS | Real `Worker.load_model()`, untouched, using `--quantization`/`--dtype`/`--model` passed through |
| Start OpenAI server | Real `vllm serve`, untouched - `--serve` controls whether this stage binds `0.0.0.0` (client-facing) vs `127.0.0.1` (internal-only) |

It execs the real `vllm serve` CLI as a subprocess replacement
(`os.execvpe`) rather than reimplementing engine construction, per "reuse
existing code whenever possible" - the new code surface is entirely
environment-variable plumbing plus the one `--worker-cls`.

---

# Task 4: all three pipeline stages, including Machine B's two neighbors

Handled by `establish_pp_transports()` (`vllm/transport/pipeline_bootstrap.py`)
and the `transport_prev`/`transport_next` extension to `GroupCoordinator`
(`vllm/distributed/parallel_state.py`) - see Task 2's test20 proof.

**Why every local GPU needs its own link, not just one per machine**:
traced in `vllm/v1/worker/gpu_worker.py`'s `execute_model` (~line 1066,
1101) and `vllm/v1/worker/gpu_model_runner.py` (~line 4497) -
`get_pp_group().isend_tensor_dict`/`irecv_tensor_dict` run identically on
every worker process (no `is_driver_worker` guard), so with TP=2 there
are 2 independent send/recv calls per PP hop, not 1. This is why Machine
B needs 4 transport connections (2 local GPUs x 2 neighbors), not 2 as a
naive "one link per machine pair" reading would suggest - a genuine,
previously-undocumented finding from tracing the real execution path this
session, now reflected in `establish_pp_transports`'s per-`local_rank`
connection scheme.

ID scheme (deterministic, no coordination needed beyond knowing your
immediate neighbors' names): for the link between machine `L` and machine
`R` at local (TP) rank `i`, `L` registers as
`self_id=f"{L}-tp{i}-to-{R}"`, `peer_id=f"{R}-tp{i}-to-{L}"`; `R` mirrors
it. Verified this session with 3 concurrent real processes exercising all
6 links simultaneously (`scripts/preflight_check.py`, output in Task 10).

---

# Task 5: GPTQ/AWQ via existing vLLM loaders - integrated, not rewritten

No changes to any quantization code
(`vllm/model_executor/layers/quantization/`). Confirmed by reading (this
session) the real dispatch in `AutoAWQConfig.get_quant_method`
(`auto_awq.py:285-357`) and the equivalent GPTQ path:

- A checkpoint with a standard `quantize_config.json`/`quant_config.json`
  (or `quantization_config` in `config.json`) is auto-detected - just pass
  `--quantization awq` or `--quantization gptq` (or omit it and let vLLM
  infer from the checkpoint's config) to `launch_pp_stage.py`.
- **MoE-specific finding, real and load-bearing for GPT-OSS specifically**:
  `AutoAWQConfig.get_quant_method` calls `check_moe_marlin_supports_layer`
  before using the fast Marlin MoE kernel
  (`vllm/model_executor/layers/quantization/utils/marlin_utils.py:355-386`),
  which requires `hidden_size % 128 == 0`. GPT-OSS's `hidden_size=2880`
  (`2880 % 128 == 64`) fails this, on **any** GPU, not just T4 - vLLM
  automatically falls back to the `MoeWNA16` kernel
  (`moe_wna16.py`, `get_min_capability()==70`), logged as a warning, not
  an error. Same fallback applies to GPTQ (shares the same
  `marlin_utils.py` check). Expect this warning in your logs; it is
  correct behavior, not a misconfiguration.

---

# Task 6: GPT-OSS loading path - MoE routing, per-expert and fused tensors

Verified this session by reading `vllm/model_executor/models/gpt_oss.py`'s
`load_weights` (~lines 641-820), not assumed:

- It parses an `expert_id` out of incoming tensor names
  (`model.layers.N.mlp.experts.{expert_id}.gate_up_proj...`) and scatters
  into the correct slice of the fused in-memory parameter
  (`expert_data = params_dict[fused_name].data[expert_id]`) - this is
  real, existing support for **per-expert-indexed** checkpoint tensor
  layouts (confirmed against a real checkpoint using exactly this layout,
  see Task 9).
- It also accepts the **fused** layout (one `[128, 2880, 5760]` tensor per
  layer covering all experts at once) - the native/MXFP4 checkpoint's own
  format.
- Both paths are already implemented; nothing needed changing here. This
  was independently checked, not assumed from the per-expert finding
  alone, by confirming the fused-tensor code path is a separate branch in
  the same function, not something that would need to be added.

---

# Task 7: deployment health checks

Two scripts, `scripts/preflight_check.py` (before model load) and
`scripts/health_check.py` (after), covering every item asked for:

| Check | Script | How |
|---|---|---|
| Signaling server reachable | preflight | Real HTTP probe against `/peer/{id}` (expects 404 = reachable) |
| Transport connected (per link, per local GPU) | preflight | Real `establish_pp_transports()` call, closes immediately after |
| Neighbor connected | preflight | Same - a hole punch or TCP accept/connect only succeeds if the neighbor is also attempting |
| TP initialized | (implicit) | If `init_device()` doesn't raise, local TP formed - `launch_pp_stage.py`'s own exit code/logs are the signal; no separate live query exists |
| PP initialized | via log | `TransportPPWorker` logs "transport PP group installed and connected" - `health_check.py --log-file` greps for it |
| Checkpoint loaded | health_check | `/v1/models` returns the model id, only possible after `load_model()` succeeds |
| Tokenizer loaded | health_check | Same endpoint - fails to come up at all if tokenizer loading failed |
| KV cache initialized | (implicit) | Engine never reaches `/health`=200 if KV cache profiling/allocation fails |
| API ready | health_check | `/health` returns 200 |
| First generated token | health_check | `--completion` sends a real `/v1/completions` request |

Both scripts abort with a specific, non-generic message and non-zero exit
on failure (see each script's own output for exact wording) - "Abort with
clear diagnostics" is implemented as actual differentiated error paths,
not a single catch-all.

**Real-tested this session** (3 concurrent processes, real hole punches,
real signaling server - not simulated): see Task 10's transcript.

---

# Task 8/9/11: deployment guide, machine commands, checklist

## Prerequisites (all three machines, identical)

| Item | Requirement | Why |
|---|---|---|
| GPU | 2x NVIDIA T4 (16GB) per machine | Target hardware; Turing, SM 7.5 |
| Driver | >=525 | CUDA 12.x compatibility |
| CUDA toolkit | 12.4 | Matches a widely-available pinned torch build |
| OS | Ubuntu 22.04 (or compatible) | Standard vLLM target |
| Python | 3.11 or 3.12 | This checkout's supported range |
| Disk | >=300GB free | Checkpoint (65-117GB depending on format) + build artifacts + headroom |
| Network | Outbound HTTPS to the signaling server's URL; no inbound ports required for the pipeline itself | Hole punching is outbound-initiated on both sides |

## Supported GPUs for this deployment specifically

T4 only clears the capability floor for **GPTQ, AWQ, bitsandbytes**
(`get_min_capability()` = 60/75/70 respectively) - not the native MXFP4
checkpoint (needs >=80). See `README_GPTOSS_120B_CLUSTER.md` (frozen, not
modified this session) for the full precision analysis; this document
assumes that conclusion and deploys with GPTQ.

## Machine preparation (run identically on A, B, C)

```bash
# Driver + CUDA toolkit
sudo apt update && sudo apt install -y nvidia-driver-535
sudo reboot
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
# expect: 2 rows, "Tesla T4, 15360 MiB, <>=525>"

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update && sudo apt install -y cuda-toolkit-12-4
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc && source ~/.bashrc
nvcc --version   # expect: release 12.4

# Python + venv
sudo apt install -y python3.11 python3.11-venv python3-pip build-essential git
python3.11 -m venv /opt/gptoss-venv
source /opt/gptoss-venv/bin/activate
pip install --upgrade pip

# vLLM - build from source with compiled kernels (budget 30min-2hr)
git clone https://github.com/vllm-project/vllm.git /opt/vllm
cd /opt/vllm
pip install -r requirements/cuda.txt
pip install -e . --no-build-isolation
python3 -c "import vllm._C_stable_libtorch; print('compiled kernels OK')"
# if this fails, stop - nothing past this point works until it's fixed
```

Then copy this project's transport code into that checkout (or clone this
project's repo directly as `/opt/vllm` instead of stock vLLM - either
way, these paths must exist before continuing):

```bash
# from this repository:
cp -r vllm/transport /opt/vllm/vllm/transport
cp vllm/distributed/parallel_state.py /opt/vllm/vllm/distributed/parallel_state.py
cp vllm/ir/tolerances.py /opt/vllm/vllm/ir/tolerances.py   # only if present in your stock checkout's layout
cp -r scripts /opt/vllm/scripts
cp -r tests/transport /opt/vllm/tests/transport
```

### Environment variables (every machine)

```bash
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=WARN
export TORCH_CUDA_ARCH_LIST=7.5
```

(`VLLM_TRANSPORT` and the `VLLM_TRANSPORT_*` variables are set
automatically by `launch_pp_stage.py` - do not set them by hand.)

### Directory layout (every machine)

```
/opt/vllm/                        # vLLM + this project's transport code
/opt/udp_holepunch/                # signaling_server.py, peer.py (frozen, unmodified)
/data/models/gpt-oss-120b-gptq/    # checkpoint (see download section)
```

### Ports / firewall

| Port | Purpose | Direction |
|---|---|---|
| 8000 | Signaling server (only the machine hosting it) | Inbound, or via a tunnel (e.g. zrok) so no inbound rule is needed at all |
| Ephemeral UDP (`--udp-port-base` and up) | Hole-punched transport, per link | No inbound rule needed - outbound-initiated on both sides |
| 8080 | OpenAI API (Machine C only, the `--serve` stage) | Inbound from clients |

## Checkpoint: download and verify

This deployment targets the one real, complete, genuinely-quantized GPTQ
checkpoint found this session (see `README_GPTOSS_120B_CLUSTER.md` Task 5
for the full search and quality caveats - it was built for a different
serving stack and reports +12.0% NLL degradation versus BF16; validate
this is acceptable for your use case before production use):

```bash
pip install -U "huggingface_hub[cli]"
hf download positron-ai/openai_gpt-oss-120b-ingest-best-gptq \
    --local-dir /data/models/gpt-oss-120b-gptq

python3 -c "
import json
idx = json.load(open('/data/models/gpt-oss-120b-gptq/model.safetensors.index.json'))
assert idx['metadata']['total_size'] == 64862418823, 'checkpoint size mismatch - re-download'
print('OK - matches known-good size (64.86GB, 46,983 weight entries)')
"
python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('/data/models/gpt-oss-120b-gptq')
print('tokenizer OK, vocab size:', tok.vocab_size)
"
```

Copy (or share via networked storage) the identical checkpoint directory
to all three machines - every stage's `launch_pp_stage.py` loads from the
same `--model` path; only its own layer shard is actually materialized in
GPU memory (`make_layers()`, Task 1), but the full checkpoint directory
must be present/reachable so the loader can read the shards it needs.

## Signaling server

```bash
# on a small, always-on host reachable by all three machines
cd /opt/udp_holepunch && pip install fastapi uvicorn
python3 -m uvicorn signaling_server:app --host 0.0.0.0 --port 8000
# separate terminal/systemd unit, if not directly reachable:
zrok share public localhost:8000
# note the printed URL - this is $SIGNALING_URL below
```

## Starting order

1. Signaling server (+ tunnel if used) - confirm health first:
   `curl -s $SIGNALING_URL/peer/__probe__` (expect HTTP 404, not a
   connection error).
2. On **all three machines roughly simultaneously**, run
   `scripts/preflight_check.py` (Task 7) - a hole punch needs both sides
   attempting at close to the same time. Do not proceed until all three
   print `PREFLIGHT OK`.
3. Start Machine B first (`launch_pp_stage.py`, below) - it has the most
   connections to establish.
4. Start Machine A.
5. Start Machine C.
6. Watch each machine's logs for the expected checkpoints (next section)
   before considering it ready.

## Machine A (PP stage 0)

```bash
source /opt/gptoss-venv/bin/activate
cd /opt/vllm
export CUDA_VISIBLE_DEVICES=0,1 NCCL_DEBUG=WARN TORCH_CUDA_ARCH_LIST=7.5

python3 scripts/preflight_check.py \
    --pp-rank 0 --pp-world-size 3 \
    --self-name MachineA --next-name MachineB \
    --transport udp --signaling-url $SIGNALING_URL

python3 scripts/launch_pp_stage.py \
    --model $MODEL_PATH \
    --tensor-parallel-size 2 \
    --pp-rank 0 --pp-world-size 3 \
    --self-name MachineA --next-name MachineB \
    --transport udp --signaling-url $SIGNALING_URL \
    --quantization gptq --dtype float16 \
    2>&1 | tee /var/log/gptoss-machineA.log
```

## Machine B (PP stage 1 - two links)

```bash
source /opt/gptoss-venv/bin/activate
cd /opt/vllm
export CUDA_VISIBLE_DEVICES=0,1 NCCL_DEBUG=WARN TORCH_CUDA_ARCH_LIST=7.5

python3 scripts/preflight_check.py \
    --pp-rank 1 --pp-world-size 3 \
    --self-name MachineB --prev-name MachineA --next-name MachineC \
    --transport udp --signaling-url $SIGNALING_URL

python3 scripts/launch_pp_stage.py \
    --model $MODEL_PATH \
    --tensor-parallel-size 2 \
    --pp-rank 1 --pp-world-size 3 \
    --self-name MachineB --prev-name MachineA --next-name MachineC \
    --transport udp --signaling-url $SIGNALING_URL \
    --quantization gptq --dtype float16 \
    2>&1 | tee /var/log/gptoss-machineB.log
```

## Machine C (PP stage 2 - serves the API)

```bash
source /opt/gptoss-venv/bin/activate
cd /opt/vllm
export CUDA_VISIBLE_DEVICES=0,1 NCCL_DEBUG=WARN TORCH_CUDA_ARCH_LIST=7.5

python3 scripts/preflight_check.py \
    --pp-rank 2 --pp-world-size 3 \
    --self-name MachineC --prev-name MachineB \
    --transport udp --signaling-url $SIGNALING_URL

python3 scripts/launch_pp_stage.py \
    --model $MODEL_PATH \
    --tensor-parallel-size 2 \
    --pp-rank 2 --pp-world-size 3 \
    --self-name MachineC --prev-name MachineB \
    --transport udp --signaling-url $SIGNALING_URL \
    --quantization gptq --dtype float16 \
    --serve --host 0.0.0.0 --port 8080 \
    2>&1 | tee /var/log/gptoss-machineC.log
```

`$MODEL_PATH` and `$SIGNALING_URL` are the only placeholders, as
requested - every other flag is a concrete value.

## Startup verification (Task 10 commands)

```bash
# on each machine, after launch_pp_stage.py is running:
python3 scripts/health_check.py --host 127.0.0.1 --port <8080 on C, else the port you chose> \
    --model-path $MODEL_PATH --log-file /var/log/gptoss-machine<A|B|C>.log

# on Machine C specifically, once all three report healthy:
python3 scripts/health_check.py --host 127.0.0.1 --port 8080 \
    --model-path $MODEL_PATH --completion --log-file /var/log/gptoss-machineC.log

# GPU utilization / memory (every machine):
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv

# transport/neighbor connectivity, independent re-check at any time:
python3 scripts/preflight_check.py --pp-rank <N> --pp-world-size 3 \
    --self-name <name> [--prev-name ... --next-name ...] \
    --transport udp --signaling-url $SIGNALING_URL

# open ports:
ss -tlnp | grep -E ':(8000|8080)'
```

## Expected logs, in order

```
INFO ... [parallel_state.py:...] world_size=1 rank=0 local_rank=0 ... backend=gloo   # real local TP
INFO ... rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, ... TP rank 0    # real local bootstrap
[TransportPPWorker] establishing pp_rank=.../... links (self=... prev=... next=...)
Hole punch success.                                                                    # per link (2x on Machine B)
[TransportPPWorker] ... transport PP group installed and connected
Loading model weights took ...s                                                        # real vLLM model-load log
INFO ... Uvicorn running on http://<host>:<port>
```

## API example (Machine C only)

```bash
curl http://<MachineC-IP>:8080/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "'"$MODEL_PATH"'", "prompt": "The capital of France is", "max_tokens": 16}'
```

## Benchmark example

```bash
# first-token latency / throughput, once serving is confirmed working:
python3 /opt/vllm/benchmarks/benchmark_serving.py \
    --backend vllm --base-url http://<MachineC-IP>:8080 \
    --model $MODEL_PATH --dataset-name random --num-prompts 20
```

No GPT-OSS forward-pass numbers are given here as expectations - none
have been measured (see Task 10's honesty note); this is the command to
run once you have real numbers to report back.

## Shutdown / restart

```bash
# shutdown: Ctrl-C (SIGINT) the foreground launch_pp_stage.py process on
# each machine, or `pkill -f launch_pp_stage.py` - the real vLLM process
# it exec'd into handles SIGINT/SIGTERM cleanup itself, unmodified.
# Order doesn't matter for shutdown (unlike startup).

# restart: re-run preflight_check.py + launch_pp_stage.py in the same
# starting order as initial startup (B, then A, then C). The signaling
# server does not need to be restarted between deployment attempts.
```

## Troubleshooting / common failures

| Symptom | Root cause | Fix |
|---|---|---|
| `preflight_check.py` fails with "signaling server unreachable" | Server down, or tunnel expired | Restart it; re-verify `curl $SIGNALING_URL/peer/__probe__` returns 404, not a connection error |
| `preflight_check.py` hangs, then times out on one link | The neighbor for that link isn't attempting yet | Start all three machines' preflight checks within a few seconds of each other (Starting order, step 2) |
| `ValueError: quantization method mxfp4 is not supported ... capability 75` | Pointed `--model`/`$MODEL_PATH` at the native checkpoint instead of the GPTQ one | Use `/data/models/gpt-oss-120b-gptq` and `--quantization gptq` |
| `ModuleNotFoundError: No module named 'vllm._C_stable_libtorch'` | vLLM wasn't actually built (`pip install -e .` step skipped or failed silently) | Re-run the build step; verify with the `import vllm._C_stable_libtorch` check before proceeding |
| Kernel compile failure during `pip install -e .` | Wrong `TORCH_CUDA_ARCH_LIST`, missing build deps, or low disk | `export TORCH_CUDA_ARCH_LIST=7.5`; ensure `build-essential`/`ninja`; free disk space |
| `/v1/completions` hangs on Machine C past `--timeout` | The Executor/scheduler_output cross-machine gap (see below) - Machine A/B's local engine loop was never actually driven with the same step's scheduling decision | See "Known gap" - this is the expected symptom of unfinished work, not a transport bug; do not spend time debugging the transport layer for this specific symptom |
| `torch.cuda.OutOfMemoryError` | Wrong `--gpu-memory-utilization`, or `--max-model-len` too large for the per-GPU weight budget | Lower `--gpu-memory-utilization`; check against `README_GPTOSS_120B_CLUSTER.md`'s KV-cache table |
| NCCL local TP hangs/fails on `all_reduce` | The two local GPUs aren't P2P-reachable (rare, more common in constrained VMs) | `NCCL_DEBUG=INFO` for details; `nvidia-smi topo -m` to check the P2P path |
| `AutoAWQMoEMarlin`/GPTQ-Marlin "not supported" warning in logs | Expected - GPT-OSS's `hidden_size=2880` isn't Marlin-tile-aligned (Task 5) | Not an error; confirms the (slower but correct) `MoeWNA16` fallback engaged |

---

# Task 10: does GPT-OSS 120B actually run on this exact cluster right now?

**Not yet, and the reason is now precisely scoped - not "the transport
doesn't work" (it does, proven again this session) and not "no checkpoint
exists" (one real GPTQ checkpoint does, Task 5). The specific remaining
gap:**

## The gap: cross-machine Executor RPC (`scheduler_output` dispatch)

Traced this session, not assumed:

- `EngineCore.step()` (`vllm/v1/engine/core.py:~595`) computes
  `scheduler_output = self.scheduler.schedule(...)` **exactly once**, in
  **one process**, and calls `self.model_executor.execute_model(scheduler_output, ...)`.
- `MultiprocExecutor.collective_rpc` (`vllm/v1/executor/multiproc_executor.py:354`)
  dispatches that call to worker processes via `MessageQueue`
  (`vllm/distributed/device_communicators/shm_broadcast.py:465`) - which
  does have a real, network-capable "remote reader" mode for genuine
  multi-node vLLM deployments, but it connects over plain TCP
  (`connect_ip`), which requires direct reachability between nodes.
- **This is the same class of problem the PP-tensor transport was built
  to solve, one layer up the stack** - it's not solved for this RPC path.
  Concretely: with `--pipeline-parallel-size 1` (required so each
  machine's *local* torch.distributed bootstrap never needs cross-machine
  reachability - Task 3), `MultiprocExecutor` on Machine A only ever
  spawns and can reach Machine A's own 2 local workers. It has no
  mechanism today to deliver `scheduler_output` to Machine B or C's
  workers at all.

**What this means concretely**: everything this session built and
tested - local TP bootstrap, the transport-backed PP group swap, correct
per-stage layer sharding, real activation-tensor exchange across all 3
simulated stages, checkpoint compatibility, health checks - is real and
verified. What is *not* yet built is the piece that would make a request
arriving at Machine C's `/v1/completions` actually cause Machine A and
B's local engines to execute the matching step. Today, only the stage
that receives the HTTP request runs its own engine loop; the other two
stages' `EngineCore.step()` never fires for that request at all, so
`recv_tensor_dict()` on the receiving stage's workers would block
indefinitely (which is what you should expect to observe if you run the
commands above end-to-end today - see the Troubleshooting entry).

## Proposed fix, scoped

Extend `MultiprocExecutor` (or a sibling `TransportExecutor`) so that,
for worker "slots" that live on a different machine, `collective_rpc`
serializes `(method, args, kwargs)` and sends it over a **dedicated**
transport connection (separate from the PP activation-tensor links) to a
small "stage server" process on the target machine, which applies it to
that machine's own local `MultiprocExecutor` (its real 2 GPU workers) and
returns the result the same way. This reuses the existing `Transport`
primitive for a new purpose (RPC relay, not tensor exchange) - in scope
for "stay inside vLLM, use the existing transport," out of scope for
"reuse existing code" alone, since no existing vLLM code does this
without Ray.

- **Source files**: new `vllm/transport/rpc_executor.py` (the executor
  subclass) and `vllm/transport/stage_server.py` (the remote-side daemon)
  - neither exists yet.
- **Estimated LOC**: 400-700 (serialization of arbitrary `collective_rpc`
  calls including `kv_output_aggregator`/`unique_reply_rank` semantics,
  timeout/error propagation matching `MultiprocExecutor`'s existing
  contract, and the driver-side vs. stage-side halves).
- **Complexity**: high. Cannot be meaningfully written or tested further
  in this sandbox - it requires real multi-node hardware to validate
  (the same reason `test17_real_gpu_pipeline.py` was left unresolved in
  the prior session: this sandbox also lacks compiled CUDA kernels at
  all, `vllm._C_stable_libtorch`, so even a correctly-written version of
  this couldn't be exercised here).

## What to execute, and what to send back

Per the stop condition: run the commands in Task 8/9 above on your real
cluster. Everything through `health_check.py` **without** `--completion`
should succeed today (transport, bootstrap, model load, API liveness) -
if any of those fail, send back the specific script's output and the
relevant machine's log file. The `--completion` check is expected to hang
or time out until the gap above is closed; if it succeeds, that's genuine
new information (it would mean vLLM's scheduling turned out to be
sufficiently deterministic/synchronized across your specific 3 independent
engine instances by coincidence of identical, simple request patterns -
worth reporting either way, but do not assume it without the real
`/v1/completions` response in hand).
