#!/usr/bin/env bash
# One-shot, cold-start-to-serving deploy for Qwen3.5-122B-A10B-GPTQ-Int4
# across exactly 3 machines (TP=2/PP=3, custom UDP hole-punch transport,
# NO MTP speculative decoding, NO Machine D) - this local sandbox
# (Machine A, stage 0) plus two remote machines B/C (stage 1, stage
# 2/driver). For whenever only 2 remote machines are available (the full
# 4-machine/MTP cluster's bring-up is manual - see docs/DEPLOYMENT.md):
# 48 layers split evenly 16/16/16, Machine C is both the last stage and
# the driver serving the API.
#
# Usage (run from the repo root):
#   1. cp .env.example .env, fill in the CURRENT session's Machine B/C SSH
#      tunnel port + root password (Machine A is this sandbox itself - no
#      credentials needed). MACHINE_D_* in .env is ignored by this script.
#   2. ./cluster/qwen35_122ba10b_3machine.sh
#
# When it finishes, the OpenAI-compatible API is live on Machine C's
# port 8080. Idempotent - every step checks current state first and skips
# what's already done, safe to re-run after a partial failure.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Real bug hit running this for real: nothing stopped two instances of
# this script from running concurrently (e.g. an automated retry/cron
# firing again before a slow first run finished) - both raced through the
# same launch/cleanup steps against the same 3 machines, corrupting the
# deployment. Take an flock lock for the whole script; a second concurrent
# invocation exits immediately instead of racing.
LOCKFILE="/tmp/qwen35_122ba10b_3machine.lock"
exec 200>"$LOCKFILE"
if ! flock -n 200; then
  echo "[3machine] ERROR: another instance of this script is already running (lock: $LOCKFILE) - refusing to start a second one." >&2
  exit 1
fi

# Real bug hit running this for real: a fresh torch/vllm install's .so
# files exist but dlopen can't find libtorch.so/libcudart.so without this -
# `import vllm._C_stable_libtorch` fails with `ImportError: libtorch.so:
# cannot open shared object file` otherwise. Must be set before any local
# vllm import below.
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
NUM_GPU_BLOCKS_OVERRIDE=60
MAX_NUM_SEQS=8

log() { echo "[3machine] $*" >&2; }

# ---- 0. load .env ----
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

# NOTE: must import torch BEFORE vllm._C_stable_libtorch - vLLM lazy-loads
# torch, so `import vllm._C_stable_libtorch` alone can't resolve
# libtorch.so via the loader even on a genuinely working install (false-
# negative crash hit running this for real). LD_LIBRARY_PATH above already
# covers this too; the explicit `import torch` first is belt-and-suspenders.
if ! python3 -c 'import torch; import vllm._C_stable_libtorch' >/dev/null 2>&1; then
  log "installing vllm (VLLM_USE_PRECOMPILED=1)..."
  pip install setuptools_rust setuptools_scm 2>&1 | tail -5
  timeout 480 env VLLM_USE_PRECOMPILED=1 pip install -e . --no-build-isolation --no-deps 2>&1 | tail -30
  timeout 900 env VLLM_USE_PRECOMPILED=1 pip install -e . --no-build-isolation 2>&1 | tail -40
  python3 -c 'import torch; import vllm._C_stable_libtorch; print("vllm compiled kernels: OK")'
else
  log "vllm compiled kernels already OK"
fi

# Real bug hit running this for real (a fresh Kaggle base image, not seen
# on every prior box this project has used): the vllm/torch install above
# upgrades numpy well past what the base image's PRE-INSTALLED
# scikit-learn was compiled against - `ValueError: numpy.dtype size
# changed, may indicate binary incompatibility` the moment anything
# imports sklearn. We don't use sklearn ourselves; `transformers`
# incidentally imports it (generation/candidate_generator.py, itself only
# reachable via a lazy AutoConfig/AutoImageProcessor import chain some
# checkpoints' config.json trigger) and that's enough to kill
# `stage_server.py`/`launch_pp_stage.py` outright before it ever attempts
# its PP transport connect - looks like a transport/timing bug (the other
# stages just hang waiting for a peer that never comes up) but isn't.
# Upgrading scikit-learn to match the now-current numpy fixes it; harmless
# no-op if it's already compatible.
if ! python3 -c 'from sklearn.metrics import roc_curve' >/dev/null 2>&1; then
  log "scikit-learn is ABI-incompatible with the numpy vllm/torch installed - upgrading..."
  pip install --upgrade scikit-learn 2>&1 | tail -15
  python3 -c 'from sklearn.metrics import roc_curve; print("scikit-learn/numpy compatible: OK")'
else
  log "scikit-learn/numpy already compatible"
fi

# Same class of bug, same fresh-image root cause: vLLM's cuda_communicator
# unconditionally imports flashinfer (for a CUSTOM_ALLREDUCE path, not
# actually load-bearing for our TP=2/T4 setup, but imported eagerly
# regardless), which chains into nvidia_cutlass_dsl's optional JAX
# integration - and the base image's pre-installed jax predates
# `jax.numpy.float8_e8m0fnu`, so importing it raises
# `AttributeError: module 'jax.numpy' has no attribute 'float8_e8m0fnu'`
# and kills the worker process during distributed-group init, again
# before the PP transport connect is ever attempted. Upgrading jax fixes
# it; harmless no-op if already compatible.
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
  log "ERROR: could not determine public signaling URL from /tmp/zrok_signaling.log (any HTTP tunnel works - zrok is this project's default, swap for ngrok/cloudflared/ssh -R if preferred)"
  exit 1
fi
log "signaling URL: $SIGNALING_URL"
curl -s -m 10 "$SIGNALING_URL/docs" -o /dev/null -w "[3machine] public signaling URL check: HTTP %{http_code}\n" || true

if [[ ! -f /data/stage0-checkpoint/model.safetensors.index.json ]]; then
  log "stage0-checkpoint missing, selectively downloading + extracting (layers 0-15 + globals)..."
  mkdir -p /data/models
  (cd humming_fix/single_layer_probe && python3 download_and_extract_qwen35_stage.py --start 0 --end 16 --out /data/stage0-checkpoint --checkpoint-dir "$CHECKPOINT_DIR" --include-globals 2>&1 | tail -40)
  log "stage0-checkpoint ready, downloaded shards deleted"
else
  log "stage0-checkpoint already present"
fi

# ---- 2-3. remote machines B, C ----
# Real bug hit running this for real: a session timeout/restart can wipe a
# remote machine's ENTIRE disk (not just /data) - torch/vllm/humming pip
# installs gone, not just the checkpoint, sometimes even a fresh host key
# (genuinely a different machine under the same name). Always run the full
# default stage set, never skip based on a stale "it was fine before"
# assumption - each stage's own check makes this fast/no-op when the
# machine really did persist.
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

# ---- 4. clean up any stale processes from a prior launch attempt ----
# A stray stage_server.py/launch_pp_stage.py (port 8080 still bound on
# Machine C, or an orphaned VLLM::Worker child still holding GPU memory -
# killing a parent does NOT kill its multiprocessing workers) makes the
# fresh launch below fail in a way that looks like a transport/timing bug
# but isn't. Clean every machine unconditionally before launching; harmless
# no-op on a genuinely fresh start.
#
# Each pattern gets its OWN pkill invocation - a single
# `pkill -f 'pat1|pat2'` matches its own argv (which literally contains the
# pattern text) and can SIGKILL itself mid-scan before finishing, silently
# leaving orphaned GPU-memory-holding workers behind on a "successful"
# cleanup (real bug hit running this for real).
log "=== cleaning up any stale processes from a prior launch attempt ==="
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

# ---- 5. launch all 3 stages together ----
# Critical: launched within seconds of each other, NOT waiting for one to
# report "ready" before starting the next - stage_server.py's PP transport
# connect and the driver's RPC connect both block waiting for their peer,
# so starting them staggered causes real connect-timeout crashes.
log "=== launching 3-stage deployment (CUDA graph, no MTP - startup takes ~8-10min for torch.compile + capture) ==="
rm -f deploy_A.log
nohup python3 -u scripts/stage_server.py \
  --model /data/stage0-checkpoint --tensor-parallel-size 2 --pp-rank 0 --pp-world-size 3 \
  --self-name MachineA --next-name MachineB --driver-name MachineC \
  --transport udp --signaling-url "$SIGNALING_URL" --transport-connect-timeout 900 \
  --quantization gptq --dtype float16 --language-model-only --max-model-len $MAX_MODEL_LEN \
  --gpu-memory-utilization $GPU_MEM_UTIL --num-gpu-blocks-override $NUM_GPU_BLOCKS_OVERRIDE \
  --max-num-seqs $MAX_NUM_SEQS --enable-cudagraph > deploy_A.log 2>&1 &

sshpass -p "$MACHINE_B_PASSWORD" ssh -o StrictHostKeyChecking=no -p "$MACHINE_B_PORT" root@127.0.0.1 \
  "cd /kaggle/working/vllm && nohup python3 -u scripts/stage_server.py \
  --model /data/stage1-checkpoint --tensor-parallel-size 2 --pp-rank 1 --pp-world-size 3 \
  --self-name MachineB --prev-name MachineA --next-name MachineC --driver-name MachineC \
  --transport udp --signaling-url $SIGNALING_URL --transport-connect-timeout 900 \
  --quantization gptq --dtype float16 --language-model-only --max-model-len $MAX_MODEL_LEN \
  --gpu-memory-utilization $GPU_MEM_UTIL --num-gpu-blocks-override $NUM_GPU_BLOCKS_OVERRIDE \
  --max-num-seqs $MAX_NUM_SEQS --enable-cudagraph \
  > /kaggle/working/vllm/deploy_B.log 2>&1 < /dev/null & disown; echo launched" &

sshpass -p "$MACHINE_C_PASSWORD" ssh -o StrictHostKeyChecking=no -p "$MACHINE_C_PORT" root@127.0.0.1 \
  "cd /kaggle/working/vllm && nohup python3 scripts/launch_pp_stage.py \
  --pp-rank 2 --pp-world-size 3 --self-name MachineC --prev-name MachineB \
  --transport udp --signaling-url $SIGNALING_URL --transport-connect-timeout 900 \
  --model /data/stage2-checkpoint --tensor-parallel-size 2 --dtype float16 --quantization gptq \
  --gpu-memory-utilization $GPU_MEM_UTIL --language-model-only --max-model-len $MAX_MODEL_LEN \
  --num-gpu-blocks-override $NUM_GPU_BLOCKS_OVERRIDE --max-num-seqs $MAX_NUM_SEQS --enable-cudagraph \
  --serve --host 0.0.0.0 --port $DRIVER_PORT --remote-stage-names MachineA,MachineB \
  > /kaggle/working/vllm/deploy_C.log 2>&1 < /dev/null & disown; echo launched" &

wait

# ---- 6. wait for the API to come up ----
log "waiting for the API server on Machine C to report ready..."
READY=0
for i in $(seq 1 40); do
  if sshpass -p "$MACHINE_C_PASSWORD" ssh -o StrictHostKeyChecking=no -p "$MACHINE_C_PORT" root@127.0.0.1 \
      "grep -q 'Application startup complete' /kaggle/working/vllm/deploy_C.log 2>/dev/null"; then
    READY=1
    break
  fi
  if sshpass -p "$MACHINE_C_PASSWORD" ssh -o StrictHostKeyChecking=no -p "$MACHINE_C_PORT" root@127.0.0.1 \
      "grep -qi 'Traceback' /kaggle/working/vllm/deploy_C.log 2>/dev/null"; then
    log "ERROR: Machine C crashed during startup - check deploy_C.log on that machine"
    exit 1
  fi
  sleep 20
done

if [[ "$READY" -ne 1 ]]; then
  log "ERROR: timed out waiting for Machine C's API server - check deploy_A.log locally and deploy_B.log/deploy_C.log on their machines"
  exit 1
fi

sshpass -p "$MACHINE_C_PASSWORD" ssh -o StrictHostKeyChecking=no -p "$MACHINE_C_PORT" root@127.0.0.1 \
  "curl -s http://127.0.0.1:${DRIVER_PORT}/health -o /dev/null -w '[3machine] API health check: HTTP %{http_code}\n'"

log "=== DEPLOYMENT READY ==="
log "API is live on Machine C, port ${DRIVER_PORT} (OpenAI-compatible: /v1/completions, /v1/chat/completions)."
log "To reach it from here:"
log "  ssh -o StrictHostKeyChecking=no -p $MACHINE_C_PORT -L 8081:127.0.0.1:${DRIVER_PORT} -N root@127.0.0.1 &"
log "  curl http://127.0.0.1:8081/v1/completions -H 'Content-Type: application/json' -d '{\"model\": \"/data/stage2-checkpoint\", \"prompt\": \"Hello\", \"max_tokens\": 20}'"
