#!/usr/bin/env bash
# One-shot (re)deploy of Qwen3.8-27B-W4A16-AutoRound-GPTQ as a single-machine
# TP=2 OpenAI/Anthropic-compatible API server on a remote 2xT4 machine
# (this project's "Akun2"), with MTP speculative decoding, int8 per-token-head
# KV cache, and Claude-Code-compatible tool/reasoning parsers enabled.
#
# Usage (run from the repo root, or anywhere - path-independent):
#   ./cluster/qwen38_27b_akun2.sh --port 9194 --password '...' \
#       --api-key '...' [--host 127.0.0.1] [--skip-checkpoint-download]
#
# Idempotent-ish: downloads the checkpoint only if /data/qwen38-27b-gptq is
# missing, and always does a fresh server (re)launch.
#
# Real bug hit running this for real (2026-08-18): `pkill -f api_server`
# issued in the SAME ssh command string as the launch itself silently killed
# the invoking SSH session before the launch even started - `pkill -f`
# matches a process's full argv, and the argv of the very shell running
# `pkill -f api_server; ...; python3 -u -m vllm.entrypoints.openai.api_server
# ...` contains the substring "api_server" too (it's literally part of the
# command text being executed), so pkill matched and killed its own parent
# shell. Symptom: ssh returns 255 with ZERO output, no log file ever
# created. Fix: kill any stale server in its OWN separate ssh call, never
# combined with the launch command.

set -euo pipefail

PORT=""
PASSWORD=""
API_KEY=""
HOST_BIND="0.0.0.0"
SKIP_DOWNLOAD=0
HF_REPO="Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ"
CKPT_DIR="/data/qwen38-27b-gptq"
REMOTE_ROOT="/kaggle/working/vllm"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --password) PASSWORD="$2"; shift 2 ;;
    --api-key) API_KEY="$2"; shift 2 ;;
    --host) HOST_BIND="$2"; shift 2 ;;
    --skip-checkpoint-download) SKIP_DOWNLOAD=1; shift ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

[[ -n "$PORT" && -n "$PASSWORD" && -n "$API_KEY" ]] || {
  echo "need --port, --password and --api-key (see --help)" >&2
  exit 1
}

LOCKFILE="/tmp/qwen38_27b_akun2.lock"
exec 200>"$LOCKFILE"
flock -n 200 || { echo "another instance is already running (lock: $LOCKFILE)" >&2; exit 1; }

ssh_cmd() { sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p "$PORT" root@127.0.0.1 "$@"; }

log() { echo "[akun2-deploy] $*" >&2; }

if [[ "$SKIP_DOWNLOAD" -eq 0 ]]; then
  have=$(ssh_cmd "[[ -f $CKPT_DIR/config.json ]] && echo yes || echo no")
  if [[ "$have" == "yes" ]]; then
    log "checkpoint already present at $CKPT_DIR, skipping download"
  else
    log "downloading checkpoint ($HF_REPO -> $CKPT_DIR)..."
    ssh_cmd "mkdir -p /data && hf download $HF_REPO --local-dir $CKPT_DIR"
    log "checkpoint download done"
  fi
fi

# Kill any stale server in its OWN ssh call - see header comment for why
# this must never share a command string with the launch below.
log "stopping any existing server..."
ssh_cmd "pkill -f api_server" || true
sleep 3

log "launching server (TP=2, MTP on, int8 KV cache, tool/reasoning parsers)..."
ssh_cmd "cd $REMOTE_ROOT && setsid nohup python3 -u -m vllm.entrypoints.openai.api_server \
  --model $CKPT_DIR \
  --served-model-name qwen3.8-27b \
  --quantization gptq \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 150544 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  --language-model-only \
  --kv-cache-dtype int8_per_token_head \
  --speculative-config '{\"method\": \"mtp\", \"num_speculative_tokens\": 1}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --api-key $API_KEY \
  --host $HOST_BIND --port 8080 \
  > $REMOTE_ROOT/deploy_qwen38_latest.log 2>&1 < /dev/null &
sleep 2
ps aux | grep api_server | grep -v grep"

# Model loading + torch.compile warmup on T4 legitimately takes several
# minutes - poll on real health-check success, not a short fixed timeout
# (see memory/feedback_wait_on_activity_not_timeout.md).
log "waiting for health check (can take 5-15 min on T4 - warmup is slow but real, don't kill early)..."
for i in $(seq 1 90); do
  code=$(ssh_cmd "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/health 2>/dev/null" || true)
  if [[ "$code" == "200" ]]; then
    log "server is UP (health check passed after ~$((i * 10))s)"
    exit 0
  fi
  errline=$(ssh_cmd "grep -i 'traceback\|OutOfMemory\|CUDA error' $REMOTE_ROOT/deploy_qwen38_latest.log 2>/dev/null | tail -1" || true)
  if [[ -n "$errline" ]]; then
    log "ERROR detected in server log: $errline"
    log "full log: $REMOTE_ROOT/deploy_qwen38_latest.log on the remote machine"
    exit 1
  fi
  sleep 10
done

log "ERROR: server did not become healthy within 15 minutes - check $REMOTE_ROOT/deploy_qwen38_latest.log"
exit 1
