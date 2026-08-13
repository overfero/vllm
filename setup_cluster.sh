#!/usr/bin/env bash
# Idempotent full-stack restore for the Qwen3.5-122B-A10B-GPTQ-Int4
# 4-machine (TP=2/PP=4, custom UDP hole-punch transport, MTP speculative
# decoding) deployment: this local sandbox (Machine A, stage 0) plus three
# remote machines B/C/D (stage 1, stage 2, stage 3/driver).
#
# Every fresh session mints NEW SSH port/password for the remote machines
# (and sometimes wipes a remote machine's disk entirely - torch/vllm/
# humming installs, /data, gone). This script brings all four back to a
# working state with a single run per machine. Safe to re-run any time -
# every step checks current state first and skips what's already done
# (see ops/setup_machine.sh, which this script drives for B/C/D).
#
# Usage:
#   1. Copy .env.example to .env and fill in the current session's
#      MACHINE_B/C/D port+password (Machine A is this local sandbox, not
#      listed).
#   2. ./setup_cluster.sh
#
# DISK/BANDWIDTH RULES THIS SCRIPT ENFORCES:
#   - The checkpoint (78.8GB full, ~50-58GB/stage selectively) is
#     downloaded independently on EACH machine directly from HuggingFace -
#     NEVER transferred machine-to-machine over SSH.
#   - Only small things (this project's own code) are rsync'd over SSH.
#   - Only the shards each stage's layer range actually needs are ever
#     downloaded (see humming_fix/single_layer_probe/
#     download_and_extract_qwen35_stage.py) and the raw shards are deleted
#     right after extraction - never keep the raw download + the extracted
#     stage checkpoint simultaneously on the same disk.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

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

log() { echo "[setup] $*" >&2; }

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
log "=== Machine A (local, stage 0) ==="

local_torch_ver=$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || true)
if [[ "$local_torch_ver" != "$TORCH_VERSION" ]]; then
  log "installing torch==$TORCH_VERSION (was: ${local_torch_ver:-none})..."
  pip install --index-url "$TORCH_INDEX_URL" "torch==$TORCH_VERSION" 2>&1 | tail -20
else
  log "torch already at $TORCH_VERSION"
fi

if ! python3 -c 'import vllm._C_stable_libtorch' >/dev/null 2>&1; then
  log "installing vllm (VLLM_USE_PRECOMPILED=1)..."
  pip install setuptools_rust setuptools_scm 2>&1 | tail -5
  timeout 480 env VLLM_USE_PRECOMPILED=1 pip install -e . --no-build-isolation --no-deps 2>&1 | tail -30
  timeout 900 env VLLM_USE_PRECOMPILED=1 pip install -e . --no-build-isolation 2>&1 | tail -40
  python3 -c 'import vllm._C_stable_libtorch; print("vllm compiled kernels: OK")'
else
  log "vllm compiled kernels already OK"
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
curl -s -m 10 "$SIGNALING_URL/docs" -o /dev/null -w "[setup] public signaling URL check: HTTP %{http_code}\n" || true

if [[ ! -f /data/stage0-checkpoint/model.safetensors.index.json ]]; then
  log "stage0-checkpoint missing, selectively downloading + extracting (layers 0-11 + globals)..."
  mkdir -p /data/models
  (cd humming_fix/single_layer_probe && python3 download_and_extract_qwen35_stage.py --start 0 --end 12 --out /data/stage0-checkpoint --checkpoint-dir "$CHECKPOINT_DIR" --include-globals 2>&1 | tail -40)
  log "stage0-checkpoint ready, downloaded shards deleted"
else
  log "stage0-checkpoint already present"
fi

# ---- 2-4. remote machines B, C, D ----
# Real bug hit running this for real: a session timeout/restart can wipe a
# remote machine's ENTIRE disk (not just /data) - torch/vllm/humming pip
# installs gone, not just the checkpoint, sometimes even a fresh host key
# (genuinely a different machine under the same name). Always run the full
# default stage set (code,torch,vllm,humming,verify), never skip based on
# a stale "it was fine before" assumption - each stage's own check (torch
# version, `import vllm._C_stable_libtorch`, `pip show humming-kernels`)
# makes this fast/no-op when the machine really did persist.
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
  # Leftover full checkpoint from a previous run that never got cleaned up
  # - harmless to check for and remove every time, cheap no-op otherwise.
  sshpass -p "$password" ssh -o StrictHostKeyChecking=no -p "$port" root@127.0.0.1 \
    "rm -rf $CHECKPOINT_DIR" 2>/dev/null || true
}

deploy_remote_stage "MachineB" "$MACHINE_B_PORT" "$MACHINE_B_PASSWORD" 12 24 /data/stage1-checkpoint
deploy_remote_stage "MachineC" "$MACHINE_C_PORT" "$MACHINE_C_PASSWORD" 24 36 /data/stage2-checkpoint
deploy_remote_stage "MachineD" "$MACHINE_D_PORT" "$MACHINE_D_PASSWORD" 36 48 /data/stage3-checkpoint ":globals"

log "=== all 4 machines restored ==="
log "signaling URL for launch commands: $SIGNALING_URL"
log "next: launch all 4 stages - see pp_tests/README.md and pp_tests/launch/launch_machine{A,B,C,D}.sh"
log "  export SIGNALING_URL=$SIGNALING_URL"
log "  ./pp_tests/launch/launch_machineA.sh              # this sandbox"
log "  ssh machineB '... SIGNALING_URL=$SIGNALING_URL pp_tests/launch/launch_machineB.sh'"
log "  ssh machineC '... SIGNALING_URL=$SIGNALING_URL pp_tests/launch/launch_machineC.sh'"
log "  ssh machineD '... SIGNALING_URL=$SIGNALING_URL pp_tests/launch/launch_machineD.sh'   # driver, serves on :8080"
