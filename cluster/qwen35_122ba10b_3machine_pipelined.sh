#!/usr/bin/env bash
# One-shot, cold-start-to-serving deploy for Qwen3.5-122B-A10B-GPTQ-Int4
# across exactly 3 machines - MICROBATCHING CANDIDATE (2026-08-15), NOT the
# baseline. Identical topology to cluster/qwen35_122ba10b_3machine.sh
# (TP=2/PP=3, custom UDP hole-punch transport, NO MTP, NO Machine D, 48
# layers split 16/16/16, NO expert-parallel) - the ONLY difference is
# `--enable-pipelining` on every stage + `--batch-queue-size 3` on the
# driver, which together let vLLM's own `step_with_batch_queue()` actually
# run (see vllm/transport/rpc_executor.py's module docstring for exactly
# what these two flags do and why BOTH are required together).
#
# THIS IS AN EXPERIMENTAL, NOT-YET-VALIDATED CANDIDATE. A previous, less
# careful attempt at this same optimization was reverted after hitting an
# undiagnosed deadlock - the wire-protocol/config changes here were
# specifically designed to close the known failure modes (see
# rpc_executor.py), but genuine correctness (not just "does it start
# without crashing") has NOT been empirically verified yet. Test with
# real completions and COMPARE TEXT against the baseline's known-good
# output before trusting this for anything - the documented failure
# signature for this class of bug is garbage output starting a few tokens
# into generation, not a crash.
#
# Baseline (PP3+TP2) and this candidate CANNOT run at the same time - both
# want ~14GiB/GPU on the same 6 T4s. This script kills any running
# deployment (baseline or a stale candidate) before launching.
#
# Usage (run from the repo root, .env already filled in - shared with the
# baseline script, no separate credentials needed):
#   ./cluster/qwen35_122ba10b_3machine_pipelined.sh
#
# When it finishes, the OpenAI-compatible API is live on Machine C's
# port 8080 (same port as baseline - they're never up simultaneously).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LOCKFILE="/tmp/qwen35_122ba10b_3machine_pipelined.lock"
exec 200>"$LOCKFILE"
if ! flock -n 200; then
  echo "[3machine-pipelined] ERROR: another instance of this script is already running (lock: $LOCKFILE) - refusing to start a second one." >&2
  exit 1
fi

export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
grep -q "LD_LIBRARY_PATH.*torch/lib" ~/.bashrc 2>/dev/null || \
  echo 'export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc

TORCH_VERSION="2.13.0+cu130"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu130"
HUMMING_KERNELS_VERSION="0.1.10"
CHECKPOINT_DIR="/data/models/qwen3.5-122b-a10b-gptq"
SIGNALING_LOCAL_PORT=8765
DRIVER_PORT=8080
MAX_MODEL_LEN=8192
GPU_MEM_UTIL=0.95
# 2026-08-15 reliability fix: this MUST be identical across all 3 machines
# (see rpc_executor.py's module docstring - scheduler_output block-table ids
# only mean the same thing if every stage sized its KV cache identically),
# but real per-machine safe capacity (vLLM's own profiler, BEFORE this
# override was applied) differs a lot - measured for real running this
# exact deployment: Machine A auto-computed 182 blocks safe, Machine B 127,
# Machine C only 37 (holds lm_head/norm as the last stage + the API server +
# RPC executor's extra threads/python state on top of the same 16 real
# layers A/B also carry - real, structural, not just noise). The old value
# here (60) was ABOVE Machine C's own auto-computed safe threshold - it was
# working in practice (real headroom observed while serving, no OOM hit),
# but riding above what vLLM's own profiler considers safe is a genuine
# latent OOM risk under peak conditions (e.g. a burst of concurrent
# near-max-length requests, or CUDA graph capture's own transient overhead)
# that just hadn't been triggered yet. 35 leaves Machine C a small margin
# below its own measured 37-block ceiling. Trade-off: total KV cache
# capacity drops from ~70k tokens (60 blocks, supports max_num_seqs=8 at
# full 8192-token requests, 8.57x concurrency headroom) to ~41k tokens (35
# blocks, ~5x concurrency headroom at full length) - fewer concurrent
# full-length requests fit, in exchange for the tightest machine no longer
# running above its own safe threshold. Revisit if this project ever
# rebalances layers away from Machine C to reclaim its headroom instead.
NUM_GPU_BLOCKS_OVERRIDE=35
MAX_NUM_SEQS=8
BATCH_QUEUE_SIZE=3

log() { echo "[3machine-pipelined] $*" >&2; }

# ---- 0. load .env (shared with the baseline script) ----
if [[ ! -f .env ]]; then
  log "ERROR: .env not found. Copy .env.example to .env and fill in the current session's MACHINE_B/C port+password first."
  exit 1
fi
set -a
source .env
set +a
: "${MACHINE_B_PORT:?.env missing MACHINE_B_PORT}" "${MACHINE_B_PASSWORD:?.env missing MACHINE_B_PASSWORD}"
: "${MACHINE_C_PORT:?.env missing MACHINE_C_PORT}" "${MACHINE_C_PASSWORD:?.env missing MACHINE_C_PASSWORD}"

# ---- 1. local Machine A (this sandbox, stage 0) ----
log "=== Machine A (local, stage 0) ==="

local_torch_ver=$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || true)
if [[ "$local_torch_ver" != "$TORCH_VERSION" ]]; then
  log "installing torch==$TORCH_VERSION (was: ${local_torch_ver:-none})..."
  pip install --index-url "$TORCH_INDEX_URL" "torch==$TORCH_VERSION" 2>&1 | tail -20
else
  log "torch already at $TORCH_VERSION"
fi

if ! python3 -c 'import torch; import vllm._C_stable_libtorch' >/dev/null 2>&1; then
  log "installing vllm (VLLM_USE_PRECOMPILED=1)..."
  pip install setuptools_rust setuptools_scm 2>&1 | tail -5
  timeout 480 env VLLM_USE_PRECOMPILED=1 pip install -e . --no-build-isolation --no-deps 2>&1 | tail -30
  timeout 900 env VLLM_USE_PRECOMPILED=1 pip install -e . --no-build-isolation 2>&1 | tail -40
  python3 -c 'import torch; import vllm._C_stable_libtorch; print("vllm compiled kernels: OK")'
else
  log "vllm compiled kernels already OK"
fi

if ! python3 -c 'from sklearn.metrics import roc_curve' >/dev/null 2>&1; then
  log "scikit-learn is ABI-incompatible with the numpy vllm/torch installed - upgrading..."
  pip install --upgrade scikit-learn 2>&1 | tail -15
  python3 -c 'from sklearn.metrics import roc_curve; print("scikit-learn/numpy compatible: OK")'
else
  log "scikit-learn/numpy already compatible"
fi

if ! python3 -c 'import jax.numpy as jnp; jnp.float8_e8m0fnu' >/dev/null 2>&1; then
  log "jax is too old for flashinfer/cutlass's optional JAX integration - upgrading..."
  pip install --upgrade jax 2>&1 | tail -15
  python3 -c 'import jax.numpy as jnp; jnp.float8_e8m0fnu; print("jax compatible: OK")'
else
  log "jax already compatible"
fi

humming_ver=$(pip show humming-kernels 2>/dev/null | grep Version || true)
if [[ "$humming_ver" != *"$HUMMING_KERNELS_VERSION"* ]]; then
  log "installing humming-kernels==$HUMMING_KERNELS_VERSION..."
  pip install "humming-kernels==$HUMMING_KERNELS_VERSION" 2>&1 | tail -10
else
  log "humming-kernels already at $HUMMING_KERNELS_VERSION"
fi

if ! curl -s -m 3 "http://127.0.0.1:${SIGNALING_LOCAL_PORT}/docs" -o /dev/null; then
  log "starting signaling server on :${SIGNALING_LOCAL_PORT}..."
  (cd udp_holepunch && nohup python3 -m uvicorn signaling_server:app --host 0.0.0.0 --port "$SIGNALING_LOCAL_PORT" > /tmp/signaling_server.log 2>&1 &)
  sleep 3
  curl -s -m 5 "http://127.0.0.1:${SIGNALING_LOCAL_PORT}/docs" -o /dev/null && log "signaling server up" || { log "ERROR: signaling server failed to start, check /tmp/signaling_server.log"; exit 1; }
else
  log "signaling server already running"
fi

if ! pgrep -f "zrok2 share public http://127.0.0.1:${SIGNALING_LOCAL_PORT}" > /dev/null; then
  log "starting public zrok tunnel for signaling server..."
  nohup zrok2 share public "http://127.0.0.1:${SIGNALING_LOCAL_PORT}" --headless > /tmp/zrok_signaling.log 2>&1 &
  sleep 8
fi
SIGNALING_URL=$(grep -oE 'https://[a-z0-9]+\.share\.zrok\.io' /tmp/zrok_signaling.log | tail -1)
if [[ -z "$SIGNALING_URL" ]]; then
  log "ERROR: could not determine public signaling URL from /tmp/zrok_signaling.log"
  exit 1
fi
log "signaling URL: $SIGNALING_URL"
curl -s -m 10 "$SIGNALING_URL/docs" -o /dev/null -w "[3machine-pipelined] public signaling URL check: HTTP %{http_code}\n" || true

if [[ ! -f /data/stage0-checkpoint/model.safetensors.index.json ]]; then
  log "stage0-checkpoint missing, selectively downloading + extracting (layers 0-15 + globals)..."
  mkdir -p /data/models
  (cd humming_fix/single_layer_probe && python3 download_and_extract_qwen35_stage.py --start 0 --end 16 --out /data/stage0-checkpoint --checkpoint-dir "$CHECKPOINT_DIR" --include-globals 2>&1 | tail -40)
  log "stage0-checkpoint ready, downloaded shards deleted"
else
  log "stage0-checkpoint already present"
fi

# ---- 2-3. remote machines B, C (same checkpoints as baseline - shared) ----
deploy_remote_stage() {
  local name="$1" port="$2" password="$3" start="$4" end="$5" out="$6" extra="${7:-}"
  log "=== $name (remote, layers [$start,$end)) ==="
  bash ops/setup_machine.sh --port "$port" --password "$password" --name "$name"
  if ! sshpass -p "$password" ssh -o StrictHostKeyChecking=no -p "$port" root@127.0.0.1 \
      "test -f $out/model.safetensors.index.json" 2>/dev/null; then
    log "$name $out missing, downloading + extracting..."
    bash ops/setup_machine.sh --port "$port" --password "$password" --name "$name" \
      --only extract --extract-stage "$start:$end:$out$extra"
  else
    log "$name $out already present"
  fi
  sshpass -p "$password" ssh -o StrictHostKeyChecking=no -p "$port" root@127.0.0.1 \
    "rm -rf $CHECKPOINT_DIR" 2>/dev/null || true
}

deploy_remote_stage "MachineB" "$MACHINE_B_PORT" "$MACHINE_B_PASSWORD" 16 32 /data/stage1-checkpoint
deploy_remote_stage "MachineC" "$MACHINE_C_PORT" "$MACHINE_C_PASSWORD" 32 48 /data/stage2-checkpoint ":globals"

log "=== all 3 machines' environments + checkpoints ready ==="

# ---- 4. clean up ANY running deployment (baseline OR a stale candidate
#      run) - PP3+TP2 and this candidate cannot coexist on the same 6 GPUs. ----
log "=== stopping any running deployment before launching microbatching candidate ==="
pkill -9 -f "scripts/stage_server.py" 2>/dev/null || true
pkill -9 -f "scripts/launch_pp_stage.py" 2>/dev/null || true
pkill -9 -f "VLLM::Worker" 2>/dev/null || true
for m in "B:$MACHINE_B_PORT:$MACHINE_B_PASSWORD" "C:$MACHINE_C_PORT:$MACHINE_C_PASSWORD"; do
  IFS=':' read -r _ port password <<< "$m"
  sshpass -p "$password" ssh -o StrictHostKeyChecking=no -p "$port" root@127.0.0.1 \
    "pkill -9 -f 'scripts/stage_server.py' 2>/dev/null; pkill -9 -f 'scripts/launch_pp_stage.py' 2>/dev/null; pkill -9 -f 'VLLM::Worker' 2>/dev/null; true" &
done
wait
sleep 3

# ---- 5. launch all 3 stages together, WITH --enable-pipelining
#      (+ --batch-queue-size on the driver only) ----
log "=== launching microbatching candidate (CUDA graph, no MTP, no EP) ==="
rm -f deploy_A_pipelined.log
nohup python3 -u scripts/stage_server.py \
  --model /data/stage0-checkpoint --tensor-parallel-size 2 --pp-rank 0 --pp-world-size 3 \
  --self-name MachineA --next-name MachineB --driver-name MachineC \
  --transport udp --signaling-url "$SIGNALING_URL" --transport-connect-timeout 900 \
  --quantization gptq --dtype float16 --language-model-only --max-model-len $MAX_MODEL_LEN \
  --gpu-memory-utilization $GPU_MEM_UTIL --num-gpu-blocks-override $NUM_GPU_BLOCKS_OVERRIDE \
  --max-num-seqs $MAX_NUM_SEQS --enable-cudagraph --enable-pipelining > deploy_A_pipelined.log 2>&1 &

sshpass -p "$MACHINE_B_PASSWORD" ssh -o StrictHostKeyChecking=no -p "$MACHINE_B_PORT" root@127.0.0.1 \
  "cd /kaggle/working/vllm && nohup python3 -u scripts/stage_server.py \
  --model /data/stage1-checkpoint --tensor-parallel-size 2 --pp-rank 1 --pp-world-size 3 \
  --self-name MachineB --prev-name MachineA --next-name MachineC --driver-name MachineC \
  --transport udp --signaling-url $SIGNALING_URL --transport-connect-timeout 900 \
  --quantization gptq --dtype float16 --language-model-only --max-model-len $MAX_MODEL_LEN \
  --gpu-memory-utilization $GPU_MEM_UTIL --num-gpu-blocks-override $NUM_GPU_BLOCKS_OVERRIDE \
  --max-num-seqs $MAX_NUM_SEQS --enable-cudagraph --enable-pipelining \
  > /kaggle/working/vllm/deploy_B_pipelined.log 2>&1 < /dev/null & disown; echo launched" &

sshpass -p "$MACHINE_C_PASSWORD" ssh -o StrictHostKeyChecking=no -p "$MACHINE_C_PORT" root@127.0.0.1 \
  "cd /kaggle/working/vllm && nohup python3 scripts/launch_pp_stage.py \
  --pp-rank 2 --pp-world-size 3 --self-name MachineC --prev-name MachineB \
  --transport udp --signaling-url $SIGNALING_URL --transport-connect-timeout 900 \
  --model /data/stage2-checkpoint --tensor-parallel-size 2 --dtype float16 --quantization gptq \
  --gpu-memory-utilization $GPU_MEM_UTIL --language-model-only --max-model-len $MAX_MODEL_LEN \
  --num-gpu-blocks-override $NUM_GPU_BLOCKS_OVERRIDE --max-num-seqs $MAX_NUM_SEQS --enable-cudagraph \
  --enable-pipelining --batch-queue-size $BATCH_QUEUE_SIZE --enable-rpc-fusion \
  --serve --host 0.0.0.0 --port $DRIVER_PORT --remote-stage-names MachineA,MachineB \
  > /kaggle/working/vllm/deploy_C_pipelined.log 2>&1 < /dev/null & disown; echo launched" &

wait

# ---- 6. wait for the API to come up ----
log "waiting for the API server on Machine C to report ready..."
READY=0
for i in $(seq 1 40); do
  if sshpass -p "$MACHINE_C_PASSWORD" ssh -o StrictHostKeyChecking=no -p "$MACHINE_C_PORT" root@127.0.0.1 \
      "grep -q 'Application startup complete' /kaggle/working/vllm/deploy_C_pipelined.log 2>/dev/null"; then
    READY=1
    break
  fi
  if sshpass -p "$MACHINE_C_PASSWORD" ssh -o StrictHostKeyChecking=no -p "$MACHINE_C_PORT" root@127.0.0.1 \
      "grep -qi 'Traceback' /kaggle/working/vllm/deploy_C_pipelined.log 2>/dev/null"; then
    log "ERROR: Machine C crashed during startup - check deploy_C_pipelined.log on that machine"
    exit 1
  fi
  sleep 20
done

if [[ "$READY" -ne 1 ]]; then
  log "ERROR: timed out waiting for Machine C's API server - check deploy_A_pipelined.log locally and deploy_B_pipelined.log/deploy_C_pipelined.log on their machines"
  exit 1
fi

sshpass -p "$MACHINE_C_PASSWORD" ssh -o StrictHostKeyChecking=no -p "$MACHINE_C_PORT" root@127.0.0.1 \
  "curl -s http://127.0.0.1:${DRIVER_PORT}/health -o /dev/null -w '[3machine-pipelined] API health check: HTTP %{http_code}\n'"

log "=== MICROBATCHING CANDIDATE DEPLOYMENT READY (UNVALIDATED - verify output correctness before trusting) ==="
log "API is live on Machine C, port ${DRIVER_PORT}."
log "To go back to the PP3+TP2 baseline: ./cluster/qwen35_122ba10b_3machine.sh"
