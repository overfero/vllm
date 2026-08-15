#!/usr/bin/env bash
# One-shot, cold-start-to-serving deploy for Qwen3.5-122B-A10B-GPTQ-Int4
# across 4 machines - MICROBATCHING + MTP CANDIDATE (2026-08-15), combining
# THREE previously-separate, individually-proven pieces that have NEVER been
# run together before:
#   1. Asymmetric 16/16/12/4 PP layer split (proven working end-to-end,
#      commit 74560238f - see docs/DEPLOYMENT.md's "Asymmetric PP splits").
#   2. MTP speculative decoding + CUDA graphs together, with
#      --cpu-offload-gb on the 16-layer stages to fit a T4's 14.56GiB
#      (proven working, commit fe3246ff6 - but that commit tested cudagraph+
#      MTP only on the UNIFORM 12/12/12/12 split, reverted FROM the
#      asymmetric split first - cudagraph+MTP+asymmetric-split TOGETHER has
#      never actually been run).
#   3. step_with_batch_queue() pipelining + RPC fusion (this project's own
#      microbatching work, cluster/qwen35_122ba10b_3machine_pipelined.sh -
#      proven on a 3-machine/no-MTP/uniform-split deployment only).
#
# THIS EXACT COMBINATION IS UNTESTED. Each piece is independently real and
# working; stacking all three is new territory - budget for a real
# correctness pass (compare output text against a known-good baseline, not
# just "does it start") the same way the pipelining candidate needed one.
#
# Topology: A=layers[0,16) (+globals), B=layers[16,32), C=layers[32,44),
# D=layers[44,48) (+globals+mtp, driver, serves the API). TP=2 per stage,
# custom UDP hole-punch transport, same as every other script in this repo.
#
# Usage (run from the repo root):
#   1. cp .env.example .env, fill in the CURRENT session's Machine B/C/D SSH
#      tunnel port + root password (Machine A is this sandbox - no creds
#      needed).
#   2. ./cluster/qwen35_122ba10b_4machine_pipelined_mtp.sh
#
# When it finishes, the OpenAI-compatible API is live on Machine D's
# port 8080.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LOCKFILE="/tmp/qwen35_122ba10b_4machine_pipelined_mtp.lock"
exec 200>"$LOCKFILE"
if ! flock -n 200; then
  echo "[4machine-pipelined-mtp] ERROR: another instance of this script is already running (lock: $LOCKFILE) - refusing to start a second one." >&2
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
MAX_NUM_SEQS=8
BATCH_QUEUE_SIZE=4

# Asymmetric split + hybrid-attention KV-cache-group fix - see
# docs/DEPLOYMENT.md's "Asymmetric PP splits" section for exactly why BOTH
# of these are required (silent NaN-output bug and an IndexError crash on
# the first real request, respectively, if either is missing on any stage).
PP_LAYER_PARTITION="16,16,12,4"
KV_CACHE_GROUP_SIZE_OVERRIDE="12"

# MTP speculative decoding - Qwen3.5's native draft head. Must be identical
# JSON on EVERY stage (including non-driver ones - see stage_server.py's
# --speculative-config docstring for the real broadcast-shape crash hit
# without it), even though the drafter module itself only ever runs on the
# driver (last PP stage).
SPECULATIVE_CONFIG='{"method": "mtp", "num_speculative_tokens": 1}'

# UNMEASURED for this topology - do not trust this number blindly.
# Every prior override value in this project (35, 37, 60...) was profiled
# for the 3-machine/16-16-16/no-MTP topology and does NOT transfer here:
# the 12- and 4-layer stages (C, D) have a completely different real VRAM
# footprint than the 16-layer stages (A, B, which also carry
# --cpu-offload-gb 3 on top). This starting value (20) is deliberately
# conservative - well under every safe threshold this project has ever
# measured on ANY topology - specifically so the FIRST run here succeeds
# and its vLLM profiler logs on all 4 machines can be read (grep
# "Overriding num_gpu_blocks" in each deploy_*.log) to derive a real,
# evidence-based number the same way 3machine_pipelined.sh's 35 was
# derived. Raise this only after reading all 4 machines' real auto-computed
# safe values and taking a margin below the tightest one.
NUM_GPU_BLOCKS_OVERRIDE=20

log() { echo "[4machine-pipelined-mtp] $*" >&2; }

# ---- 0. load .env ----
if [[ ! -f .env ]]; then
  log "ERROR: .env not found. Copy .env.example to .env and fill in the current session's MACHINE_B/C/D port+password first."
  exit 1
fi
set -a
source .env
set +a
: "${MACHINE_B_PORT:?.env missing MACHINE_B_PORT}" "${MACHINE_B_PASSWORD:?.env missing MACHINE_B_PASSWORD}"
: "${MACHINE_C_PORT:?.env missing MACHINE_C_PORT}" "${MACHINE_C_PASSWORD:?.env missing MACHINE_C_PASSWORD}"
: "${MACHINE_D_PORT:?.env missing MACHINE_D_PORT}" "${MACHINE_D_PASSWORD:?.env missing MACHINE_D_PASSWORD}"

# ---- 1. local Machine A (this sandbox, stage 0) ----
log "=== Machine A (local, stage 0, layers [0,16)) ==="

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
curl -s -m 10 "$SIGNALING_URL/docs" -o /dev/null -w "[4machine-pipelined-mtp] public signaling URL check: HTTP %{http_code}\n" || true

if [[ ! -f /data/stage0-checkpoint/model.safetensors.index.json ]]; then
  log "stage0-checkpoint missing, selectively downloading + extracting (layers 0-15 + globals)..."
  mkdir -p /data/models
  (cd humming_fix/single_layer_probe && python3 download_and_extract_qwen35_stage.py --start 0 --end 16 --out /data/stage0-checkpoint --checkpoint-dir "$CHECKPOINT_DIR" --include-globals 2>&1 | tail -40)
  log "stage0-checkpoint ready, downloaded shards deleted"
else
  log "stage0-checkpoint already present"
fi

# ---- 2-4. remote machines B, C, D ----
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
deploy_remote_stage "MachineC" "$MACHINE_C_PORT" "$MACHINE_C_PASSWORD" 32 44 /data/stage2-checkpoint
deploy_remote_stage "MachineD" "$MACHINE_D_PORT" "$MACHINE_D_PASSWORD" 44 48 /data/stage3-checkpoint ":globals:mtp"

log "=== all 4 machines' environments + checkpoints ready ==="

# ---- 5. clean up ANY running deployment (baseline, EP, pipelined-3machine,
#      or a stale run of this candidate) - none of this project's deploy
#      scripts can coexist, they all want the same 8 T4 GPUs (6 across
#      A/B/C in the 3-machine scripts, 8 across A/B/C/D here). ----
log "=== stopping any running deployment before launching 4-machine candidate ==="
pkill -9 -f "scripts/stage_server.py" 2>/dev/null || true
pkill -9 -f "scripts/launch_pp_stage.py" 2>/dev/null || true
pkill -9 -f "VLLM::Worker" 2>/dev/null || true
for m in "B:$MACHINE_B_PORT:$MACHINE_B_PASSWORD" "C:$MACHINE_C_PORT:$MACHINE_C_PASSWORD" "D:$MACHINE_D_PORT:$MACHINE_D_PASSWORD"; do
  IFS=':' read -r _ port password <<< "$m"
  sshpass -p "$password" ssh -o StrictHostKeyChecking=no -p "$port" root@127.0.0.1 \
    "pkill -9 -f 'scripts/stage_server.py' 2>/dev/null; pkill -9 -f 'scripts/launch_pp_stage.py' 2>/dev/null; pkill -9 -f 'VLLM::Worker' 2>/dev/null; true" &
done
wait
sleep 3

# ---- 6. launch all 4 stages together, WITH --enable-pipelining on every
#      stage, --batch-queue-size + --enable-rpc-fusion on the driver (D)
#      only, --speculative-config identically on all 4, --cpu-offload-gb 3
#      only on the two 16-layer stages (A, B) ----
log "=== launching 4-machine pipelined+MTP candidate (UNTESTED combination - verify real output correctness before trusting) ==="
rm -f deploy_A_4m.log
VLLM_PP_LAYER_PARTITION="$PP_LAYER_PARTITION" VLLM_KV_CACHE_GROUP_SIZE_OVERRIDE="$KV_CACHE_GROUP_SIZE_OVERRIDE" \
nohup python3 -u scripts/stage_server.py \
  --model /data/stage0-checkpoint --tensor-parallel-size 2 --pp-rank 0 --pp-world-size 4 \
  --self-name MachineA --next-name MachineB --driver-name MachineD \
  --transport udp --signaling-url "$SIGNALING_URL" --transport-connect-timeout 900 \
  --quantization gptq --dtype float16 --language-model-only --max-model-len $MAX_MODEL_LEN \
  --gpu-memory-utilization $GPU_MEM_UTIL --num-gpu-blocks-override $NUM_GPU_BLOCKS_OVERRIDE \
  --max-num-seqs $MAX_NUM_SEQS --enable-cudagraph --enable-pipelining \
  --speculative-config "$SPECULATIVE_CONFIG" --cpu-offload-gb 3 > deploy_A_4m.log 2>&1 &

sshpass -p "$MACHINE_B_PASSWORD" ssh -o StrictHostKeyChecking=no -p "$MACHINE_B_PORT" root@127.0.0.1 \
  "cd /kaggle/working/vllm && VLLM_PP_LAYER_PARTITION='$PP_LAYER_PARTITION' VLLM_KV_CACHE_GROUP_SIZE_OVERRIDE='$KV_CACHE_GROUP_SIZE_OVERRIDE' \
  nohup python3 -u scripts/stage_server.py \
  --model /data/stage1-checkpoint --tensor-parallel-size 2 --pp-rank 1 --pp-world-size 4 \
  --self-name MachineB --prev-name MachineA --next-name MachineC --driver-name MachineD \
  --transport udp --signaling-url $SIGNALING_URL --transport-connect-timeout 900 \
  --quantization gptq --dtype float16 --language-model-only --max-model-len $MAX_MODEL_LEN \
  --gpu-memory-utilization $GPU_MEM_UTIL --num-gpu-blocks-override $NUM_GPU_BLOCKS_OVERRIDE \
  --max-num-seqs $MAX_NUM_SEQS --enable-cudagraph --enable-pipelining \
  --speculative-config '$SPECULATIVE_CONFIG' --cpu-offload-gb 3 \
  > /kaggle/working/vllm/deploy_B_4m.log 2>&1 < /dev/null & disown; echo launched" &

sshpass -p "$MACHINE_C_PASSWORD" ssh -o StrictHostKeyChecking=no -p "$MACHINE_C_PORT" root@127.0.0.1 \
  "cd /kaggle/working/vllm && VLLM_PP_LAYER_PARTITION='$PP_LAYER_PARTITION' VLLM_KV_CACHE_GROUP_SIZE_OVERRIDE='$KV_CACHE_GROUP_SIZE_OVERRIDE' \
  nohup python3 -u scripts/stage_server.py \
  --model /data/stage2-checkpoint --tensor-parallel-size 2 --pp-rank 2 --pp-world-size 4 \
  --self-name MachineC --prev-name MachineB --next-name MachineD --driver-name MachineD \
  --transport udp --signaling-url $SIGNALING_URL --transport-connect-timeout 900 \
  --quantization gptq --dtype float16 --language-model-only --max-model-len $MAX_MODEL_LEN \
  --gpu-memory-utilization $GPU_MEM_UTIL --num-gpu-blocks-override $NUM_GPU_BLOCKS_OVERRIDE \
  --max-num-seqs $MAX_NUM_SEQS --enable-cudagraph --enable-pipelining \
  --speculative-config '$SPECULATIVE_CONFIG' \
  > /kaggle/working/vllm/deploy_C_4m.log 2>&1 < /dev/null & disown; echo launched" &

sshpass -p "$MACHINE_D_PASSWORD" ssh -o StrictHostKeyChecking=no -p "$MACHINE_D_PORT" root@127.0.0.1 \
  "cd /kaggle/working/vllm && VLLM_PP_LAYER_PARTITION='$PP_LAYER_PARTITION' VLLM_KV_CACHE_GROUP_SIZE_OVERRIDE='$KV_CACHE_GROUP_SIZE_OVERRIDE' \
  nohup python3 scripts/launch_pp_stage.py \
  --pp-rank 3 --pp-world-size 4 --self-name MachineD --prev-name MachineC \
  --transport udp --signaling-url $SIGNALING_URL --transport-connect-timeout 900 \
  --model /data/stage3-checkpoint --tensor-parallel-size 2 --dtype float16 --quantization gptq \
  --gpu-memory-utilization $GPU_MEM_UTIL --language-model-only --max-model-len $MAX_MODEL_LEN \
  --num-gpu-blocks-override $NUM_GPU_BLOCKS_OVERRIDE --max-num-seqs $MAX_NUM_SEQS --enable-cudagraph \
  --speculative-config '$SPECULATIVE_CONFIG' \
  --enable-pipelining --batch-queue-size $BATCH_QUEUE_SIZE --enable-rpc-fusion \
  --serve --host 0.0.0.0 --port $DRIVER_PORT --remote-stage-names MachineA,MachineB,MachineC \
  > /kaggle/working/vllm/deploy_D_4m.log 2>&1 < /dev/null & disown; echo launched" &

wait

# ---- 7. wait for the API to come up ----
log "waiting for the API server on Machine D to report ready..."
READY=0
for i in $(seq 1 40); do
  if sshpass -p "$MACHINE_D_PASSWORD" ssh -o StrictHostKeyChecking=no -p "$MACHINE_D_PORT" root@127.0.0.1 \
      "grep -q 'Application startup complete' /kaggle/working/vllm/deploy_D_4m.log 2>/dev/null"; then
    READY=1
    break
  fi
  if sshpass -p "$MACHINE_D_PASSWORD" ssh -o StrictHostKeyChecking=no -p "$MACHINE_D_PORT" root@127.0.0.1 \
      "grep -qi 'Traceback' /kaggle/working/vllm/deploy_D_4m.log 2>/dev/null"; then
    log "ERROR: Machine D crashed during startup - check deploy_D_4m.log on that machine (also check deploy_A_4m.log locally and deploy_B_4m.log/deploy_C_4m.log remotely - a CUDA OOM on A/B during weight loading would show there, not on D)"
    exit 1
  fi
  sleep 20
done

if [[ "$READY" -ne 1 ]]; then
  log "ERROR: timed out waiting for Machine D's API server - check deploy_A_4m.log locally and deploy_B_4m.log/deploy_C_4m.log/deploy_D_4m.log on their machines"
  exit 1
fi

sshpass -p "$MACHINE_D_PASSWORD" ssh -o StrictHostKeyChecking=no -p "$MACHINE_D_PORT" root@127.0.0.1 \
  "curl -s http://127.0.0.1:${DRIVER_PORT}/health -o /dev/null -w '[4machine-pipelined-mtp] API health check: HTTP %{http_code}\n'"

log "=== 4-MACHINE PIPELINED+MTP CANDIDATE DEPLOYMENT READY (UNVALIDATED - this exact 3-way combination has never been tested, verify real output correctness before trusting) ==="
log "API is live on Machine D, port ${DRIVER_PORT}."
log "NEXT STEP: grep 'Overriding num_gpu_blocks' deploy_A_4m.log (and the same on B/C/D remotely) to see each machine's real auto-computed safe block count under THIS topology, and reconsider NUM_GPU_BLOCKS_OVERRIDE=$NUM_GPU_BLOCKS_OVERRIDE (currently an unmeasured, conservative placeholder) against real numbers before relying on this for anything but a first correctness check."
