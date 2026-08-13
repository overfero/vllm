# Current deployment: Qwen3.5-122B-A10B-GPTQ-Int4, 4 machines, MTP speculative decoding

This is the current, validated state of the cluster. For the debugging
history that got here (including dead ends and fixes for issues no longer
present), see [`docs/history/`](history/).

## Model

`Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` - hybrid Gated-DeltaNet / MoE / full-attention
decoder, 48 layers, 256 routed experts + 1 shared expert/layer, GPTQ-Int4 on
the routed experts (attention, shared expert, embeddings, lm_head, and MTP
stay unquantized). Natively multimodal; served **text-only** via
`--language-model-only` (vision tower, 333 tensors, never loaded).

## Hardware and topology

4 machines, 2x Tesla T4 (16GB) each - 8 GPUs / 128GB VRAM total.
TP=2 per machine (real NCCL, local only), PP=4 across machines (this
project's UDP hole-punch transport). 12 decoder layers per stage:

| Machine | pp-rank | layers | globals | role |
|---|---|---|---|---|
| A | 0 | 0-11 | embed_tokens, norm, lm_head | first stage |
| B | 1 | 12-23 | - | middle stage |
| C | 2 | 24-35 | - | middle stage |
| D | 3 | 36-47 | embed_tokens, norm, lm_head, **MTP drafter** | driver, serves HTTP |

Checkpoint extraction is selective (`hf download --include <only the
shards a stage's layer range touches>`) - each 12-layer stage pulls
~20-26 of 39 shards (~50-58GB) instead of the full 78.8GB, keeping peak
per-machine disk footprint bounded. See
`humming_fix/single_layer_probe/download_and_extract_qwen35_stage.py`.
Extraction itself is memory-safe: source shards are processed in small
batches (`--shards-per-batch`, default 4) and written to their own part
file immediately, bounding peak RSS to ~3-6GB regardless of total stage
size - the original single-dict-then-one-save_file() approach drove RSS to
~26GB on a stage with MTP tensors included.

## Bring-up

```bash
cp .env.example .env   # fill in current session's MACHINE_B/C/D port+password
./setup_cluster.sh
export SIGNALING_URL=https://your-tunnel   # printed by setup_cluster.sh
./pp_tests/launch/launch_machineA.sh                                    # local
ssh machineB 'cd /kaggle/working/vllm && SIGNALING_URL=... pp_tests/launch/launch_machineB.sh'
ssh machineC 'cd /kaggle/working/vllm && SIGNALING_URL=... pp_tests/launch/launch_machineC.sh'
ssh machineD 'cd /kaggle/working/vllm && SIGNALING_URL=... pp_tests/launch/launch_machineD.sh'  # driver, :8080
```

Order doesn't matter much - every stage retries hole-punching against the
signaling server until its neighbors show up (`--transport-connect-timeout`,
default 300s in the launch scripts; raise it for slow/high-latency links).

## MTP (multi-token prediction / speculative decoding)

`--speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'` is
passed to **every** stage, not just the driver. Getting this working
across the synthetic multi-machine PP group (as opposed to vLLM's native
same-process PP, which is what Qwen3.5's MTP code was written against)
took three real, separate fixes:

1. **`Qwen3_5MultiTokenPredictor.forward()`'s `is_first_rank` check**
   (`vllm/model_executor/models/qwen3_5_mtp.py`) branches on
   `get_pp_group().is_first_rank` to decide whether to build its input from
   `hidden_states` directly or from `intermediate_tensors` (meant for
   real vLLM PP, where MTP is split across the same ranks as the main
   model). This project's MTP module is never split - it's always fully
   local to the driver - but `get_pp_group()` returns the real synthetic
   PP rank, which is correctly "not first rank" for the *main model's*
   partitioning but wrongly read by the MTP class as "expect
   `intermediate_tensors` from a previous rank," which are genuinely
   `None` here. Fixed with a monkeypatch,
   `vllm/transport/qwen35_mtp_pp_fix.py`, that always takes the real
   forward's `is_first_rank=True` branch.
2. **Non-driver stages need `--speculative-config` too**, even though they
   never construct MTP drafter weights (that's separately gated on
   `if self.speculative_config and get_pp_group().is_last_rank` in
   `gpu_model_runner.py`). Without it, a non-driver stage's own array
   pre-allocation is sized for 1 token/sequence/step while the driver's
   scheduler (aware of `num_speculative_tokens=1`) sends batches sized for
   2 - a local shape mismatch (`could not broadcast input array from shape
   (2,) into shape (1,)`) on the very first real generation request.
   `scripts/stage_server.py` now accepts and forwards the flag on every
   stage.
3. **The patch from (1) has to actually reach every worker process.** The
   first attempt applied it via a `sitecustomize.py` on `PYTHONPATH`, which
   works for the main process but silently never reaches the TP worker
   subprocesses: vLLM forces Python's `multiprocessing` `spawn` start
   method whenever CUDA is involved, and `spawn` clones the **parent's
   already-computed `sys.path`** into each child instead of re-deriving it
   from `PYTHONPATH` in a fresh interpreter - so mutating
   `os.environ["PYTHONPATH"]` at runtime in the parent never reaches the
   workers, confirmed by direct reproduction (a spawned child's `sys.path`
   omits any directory added to `PYTHONPATH` after the parent interpreter
   itself started). The MTP drafter's `dummy_run`/`forward` execute inside
   those worker processes, on **every** stage (not just the driver) during
   `profile_run()`, so this bug reproduced as the same
   `assert intermediate_tensors is not None` on every non-driver stage,
   independent of fix (1). Fixed for real by moving the patch import
   directly into `vllm/transport/pp_worker.py`
   (`import vllm.transport.qwen35_mtp_pp_fix` at module level) - every
   worker process genuinely re-imports that file once as part of resolving
   `--worker-cls`, regardless of spawn/fork or `PYTHONPATH`.

**Known limitation, not fixed**: `--enable-cudagraph` together with MTP hits
a real `torch.compile`/Dynamo limitation tracing the MTP drafter's forward
(`Data-dependent assertion failed (cannot compile partial graph)`). Current
deployment runs the whole pipeline in eager mode when MTP is enabled -
CUDA graphs and MTP have not been made to work together.

## Measured results

**Network latency** (`pp_tests/real_ping_pong.py`, real UDP round-trip
between actual public IPs, not relayed): co-located machines ~0.8ms RTT;
cross-continent pairs ~104ms RTT. Confirmed as the dominant cost in
per-token time budget for geographically dispersed machines.

**CUDA graphs, non-MTP** (`--enable-cudagraph`): eager-mode compute ~225ms
total across 3 stages vs. ~39ms with CUDA graphs captured (~5.8x compute
speedup) - but since network transfer time is unchanged, aggregate
throughput only improved 2.0 -> 2.8 tok/s. Network dominates, not compute,
for this cluster's real geographic dispersion.

**MTP, eager mode, 4 machines**: ~3.9 tok/s real generation (measured via
`/v1/completions`, 100 completion tokens, temperature 0), vs. ~0.6-1.4
tok/s on the equivalent non-MTP 4-stage eager baseline. Each accepted
speculative token amortizes one full network round-trip across the
pipeline, which is the whole reason MTP was worth pursuing here (compute
was never the bottleneck - round-trips were).

## Known environment gotchas (if reproducing this)

- `df -h /` on these machines does **not** reliably reflect real disk
  quota - track your own peak usage instead of trusting free-space
  numbers.
- Never transfer large (multi-GB) files machine-to-machine over SSH -
  re-download/re-derive independently on each machine instead.
- `pkill -f <pattern>` executed over SSH can match its own invoking
  command line and kill the SSH session before it echoes anything - use
  `pkill -f '[p]attern'` (bracket-escape one character) to avoid
  self-matching.
- A `nohup`'d background process, even with all three FDs redirected to a
  file, can still hold the SSH channel open because `multiprocessing`
  worker children can inherit descriptors in ways `disown` doesn't fully
  detach - if a launch command over SSH seems to hang after the actual
  work already started, check the remote process directly rather than
  assuming a real hang.
