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
project's UDP hole-punch transport). **Asymmetric** split, 16/16/12/4
decoder layers per stage (not the uniform 12/12/12/12 this project
started with - see "Asymmetric PP splits" below for what an uneven split
requires that a uniform one doesn't):

| Machine | pp-rank | layers | globals | role |
|---|---|---|---|---|
| A | 0 | 0-15 | embed_tokens, norm, lm_head | first stage |
| B | 1 | 16-31 | - | middle stage |
| C | 2 | 32-43 | - | middle stage |
| D | 3 | 44-47 | embed_tokens, norm, lm_head, **MTP drafter** | driver, serves HTTP |

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

**Real GPU-memory cost, not just correctness**: the drafter object
(`self.drafter` in `GPUModelRunner`) gets constructed on **every** stage,
not just the driver, despite `gpu_model_runner.py`'s own gating on
`get_pp_group().is_last_rank` - see "Asymmetric PP splits" below for why
(the trivial pre-swap `_PP` group at `__init__` time is always
"last rank" for a lone rank). It costs real VRAM (own vocab embedding +
one extra MoE decoder layer, ~1-2GB) on every non-driver stage even
though it's never invoked there. Tried deleting it post-hoc in
`pp_worker.py` to reclaim that memory - reverted, real crash:
`AttributeError: 'GPUModelRunner' object has no attribute 'drafter'`,
because plenty of other `GPUModelRunner` methods (`execute_model`,
`initialize_kv_cache`, `dummy_run`, ...) reference `self.drafter`
unconditionally whenever `self.speculative_config` is set, with no
`is_last_rank` guard. If a stage's real per-layer VRAM budget is tight
with this overhead present (as with the 16-layer stages below), use
`--cpu-offload-gb` instead of trying to remove the drafter.

## Asymmetric PP splits

Splitting the 48 layers unevenly across stages (this deployment's
16/16/12/4, vs. the uniform 12/12/12/12 this project started with) is
**not** just a checkpoint-extraction range change. Two real, separate
fixes were needed, both because each stage here runs as its own fully
independent vLLM engine process (only aware of its own checkpoint shard's
layers - see `vllm/transport/pipeline_bootstrap.py`), unlike real vLLM PP
where layer partitioning is coordinated once across all ranks:

1. **`VLLM_PP_LAYER_PARTITION` env var, set identically on every stage**
   (e.g. `"16,16,12,4"`). Without it, `vllm/distributed/utils.py`'s
   `get_pp_indices()` silently falls back to an *even* division
   (`num_hidden_layers // pp_size`) regardless of what layers a stage's
   checkpoint shard actually contains - every stage ends up constructing
   the WRONG layer range (missing layers left randomly-initialized),
   producing NaN/garbage output with no error, no crash - the pipeline
   connects and "works," it just generates nonsense. This is a silent
   correctness bug, not a crash - verify real generation output, not just
   that the cluster connects.
2. **`VLLM_KV_CACHE_GROUP_SIZE_OVERRIDE` env var, set identically on
   every stage** (e.g. `"12"` for this model - see below for how to
   derive it). Qwen3.5 is a hybrid full-attention/linear-attention
   (GDN) model; `vllm/v1/core/kv_cache_utils.py`'s
   `_get_kv_cache_groups_uniform_page_size()` groups layers into
   KV-cache "groups" using a `group_size` derived from
   `min(count of each layer type present)` - an **absolute per-stage
   count**, not a ratio. A uniform 12-layer split happens to give every
   stage the same type counts (3 full-attention : 9 linear-attention),
   so every stage computed the same `group_size` "by accident." An
   asymmetric split breaks this: stage A/B (16 layers) get 4 full-attn
   layers, stage C (12 layers) gets 3, stage D (4 layers) gets 1 - three
   different `group_size` values, hence three different KV-cache group
   counts. The driver's centrally-scheduled `block_ids` tuple is shaped
   for the *driver's own* group count, so applying it on a stage with a
   different group count raises `IndexError: tuple index out of range`
   in `block_table.py`'s `add_row` on the very first real generation
   request - the cluster connects fine, model loads fine, this only
   surfaces once a `/v1/completions` request actually runs. Fixed by
   forcing every stage to use the SAME `group_size`, derived once from
   the FULL 48-layer model's layer-type counts (12 full-attention : 36
   linear-attention → `group_size = min(12, 36) = 12`), not each stage's
   own local subset.

**To derive the right `VLLM_KV_CACHE_GROUP_SIZE_OVERRIDE` for a different
split or model**: count `full_attention` vs. the other type(s) across
the model's **entire** `layer_types` list (not any one stage's subset)
and take the min:
```bash
python3 -c "
import json
c = json.load(open('/data/stage0-checkpoint/config.json'))
lt = c['text_config']['layer_types']
from collections import Counter
counts = Counter(lt)
print(counts, '-> group_size =', min(counts.values()))
"
```

**GPU memory for wider stages**: a 16-layer stage's real weights (~12.5GB
per T4 after TP=2) plus the always-present MTP drafter overhead (~1-2GB,
see above) leaves too little headroom for KV cache on a 14.56GiB-usable
T4 - `CUDA out of memory` during weight loading, before KV cache
allocation is even reached (`--num-gpu-blocks-override` does NOT help,
it only affects the later KV-cache phase). Fixed with
`--cpu-offload-gb 3` on the 16-layer stages (A, B) - offloads that much
of the weights to pinned host RAM instead. Required adding `--cpu-offload-gb`
as a real passthrough flag in `scripts/stage_server.py` (not present by
default - it only forwards a curated arg allowlist to `EngineArgs`).

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
- vLLM's own worker/engine-core subprocesses rename themselves via
  `setproctitle` to things like `VLLM::Worker_PP0_TP0` / `VLLM::EngineCore`
  - `pkill -f 'stage_server.py'` or `'vllm.entrypoints'` does **not** match
  these and leaves them running (holding GPU memory) after a "clean" kill.
  Also `pkill -f '[V]LLM::'` when cleaning these up.
- After a real `kill -9` of a vLLM process, `nvidia-smi` can keep
  reporting the GPU memory as still in-use for a short while even though
  no process holds it (`fuser -v /dev/nvidia0` shows nothing) - this is
  the driver, not a real leak; poll `nvidia-smi` for a few seconds before
  concluding a kill didn't work or the machine needs a full reset.
- These SSH/zrok tunnels can drop mid-command with no warning
  (`kex_exchange_identification: read: Connection reset by peer`) even
  when the underlying machine is fine - retry the connection a few times
  before assuming the machine itself needs a rebuild.
- UDP hole-punch success between two specific machines is not always
  symmetric or fast: one side can log success and move on while the
  peer is still retrying, and a specific pair (e.g. driver ↔ one
  non-adjacent stage) can simply take much longer or fail this run even
  when other pairs succeed quickly - a failed/stuck connect is often
  worth one clean retry (relaunch just the stuck stage(s), no need to
  restart already-connected stages) before assuming a real config bug.
