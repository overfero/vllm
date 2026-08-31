#!/usr/bin/env bash
# One-shot (re)deploy of Qwen3.8-27B-W4A16-AutoRound-GPTQ as a single-machine
# TP=2 OpenAI/Anthropic-compatible API server on THIS local 2xT4 machine,
# with MTP speculative decoding, int8 per-token-head KV cache, and
# Claude-Code-compatible tool/reasoning parsers enabled.
#
# Adapted from qwen38_27b_akun2.sh for local (no-SSH) execution, single-user
# focus (max-num-seqs=1 instead of 8 - see that script's header for the real
# pkill-self-match bug this avoids by never combining kill+launch).
#
# Usage:
#   ./cluster/qwen38_27b_local.sh --api-key '...' [--port 8080] [--host 0.0.0.0] \
#       [--skip-checkpoint-download]

set -euo pipefail

PORT="8080"
API_KEY=""
HOST_BIND="0.0.0.0"
SKIP_DOWNLOAD=0
HF_REPO="Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ"
CKPT_DIR="/data/qwen38-27b-gptq"
REPO_ROOT="/kaggle/working/vllm"
VENV_DIR="${VLLM_BUILD_VENV:-/vllm_build_venv}"
LOG_FILE="$REPO_ROOT/deploy_qwen38_local.log"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --api-key) API_KEY="$2"; shift 2 ;;
    --host) HOST_BIND="$2"; shift 2 ;;
    --skip-checkpoint-download) SKIP_DOWNLOAD=1; shift ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

[[ -n "$API_KEY" ]] || { echo "need --api-key (see --help)" >&2; exit 1; }

LOCKFILE="/tmp/qwen38_27b_local.lock"
exec 200>"$LOCKFILE"
flock -n 200 || { echo "another instance is already running (lock: $LOCKFILE)" >&2; exit 1; }

log() { echo "[qwen38-local-deploy] $*" >&2; }

if [[ "$SKIP_DOWNLOAD" -eq 0 ]]; then
  if [[ -f "$CKPT_DIR/config.json" ]]; then
    log "checkpoint already present at $CKPT_DIR, skipping download"
  else
    log "downloading checkpoint ($HF_REPO -> $CKPT_DIR)..."
    mkdir -p /data
    "$VENV_DIR/bin/python3" -c "from huggingface_hub import snapshot_download; snapshot_download('$HF_REPO', local_dir='$CKPT_DIR')"
    log "checkpoint download done"
  fi
fi

# Kill any stale server FIRST, as its own step - never combined with the
# launch command below (see qwen38_27b_akun2.sh header for the real
# pkill-self-match bug this avoids: `pkill -f api_server` run in the SAME
# command string as a launch that itself contains "api_server" in its argv
# kills its own invoking shell before the launch even starts).
log "stopping any existing server..."
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 3

log "launching server (TP=2, MTP on, int8 KV cache, single-user max-num-seqs=1)..."
cd "$REPO_ROOT"
setsid nohup "$VENV_DIR/bin/python3" -u -m vllm.entrypoints.openai.api_server \
  --model "$CKPT_DIR" \
  --served-model-name qwen3.8-27b \
  --quantization gptq \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 130000 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192 \
  --language-model-only \
  --kv-cache-dtype int8_per_token_head \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --api-key "$API_KEY" \
  --host "$HOST_BIND" --port "$PORT" \
  > "$LOG_FILE" 2>&1 < /dev/null &
sleep 2
ps aux | grep api_server | grep -v grep

# Model loading + torch.compile warmup on T4 legitimately takes several
# minutes - poll on real health-check success, not a short fixed timeout
# (see memory/feedback_wait_on_activity_not_timeout.md).
log "waiting for health check (can take 5-15 min on T4 - warmup is slow but real, don't kill early)..."
for i in $(seq 1 90); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null || true)
  if [[ "$code" == "200" ]]; then
    log "server is UP (health check passed after ~$((i * 10))s)"
    exit 0
  fi
  errline=$(grep -i 'traceback\|OutOfMemory\|CUDA error' "$LOG_FILE" 2>/dev/null | grep -v 'sitecustomize\|PYTHONVERBOSE' | tail -1 || true)
  if [[ -n "$errline" ]]; then
    log "ERROR detected in server log: $errline"
    log "full log: $LOG_FILE"
    exit 1
  fi
  sleep 10
done

log "ERROR: server did not become healthy within 15 minutes - check $LOG_FILE"
exit 1
