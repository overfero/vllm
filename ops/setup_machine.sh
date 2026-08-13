#!/usr/bin/env bash
# Idempotent remote-machine bring-up for the Qwen3.5-122B-A10B-GPTQ-Int4
# 3-machine TP=2/PP=3 deployment. Run from the ORCHESTRATOR (this sandbox)
# against one target machine over SSH (sshpass password auth, matching this
# project's zrok-tunneled access pattern - each fresh Kaggle machine gets a
# new port/password from the user).
#
# Exists because manually redoing this by hand (torch upgrade, .so copy,
# vllm build, humming-kernels, checkpoint download/extraction) after every
# machine reset cost real time this session. Every stage checks current
# state first, so re-running after a partial failure or a fresh reset only
# redoes what's actually missing.
#
# Usage:
#   ./setup_machine.sh --port 9195 --password '...' --name akun5
#   ./setup_machine.sh --port 9195 --password '...' --name akun5 \
#       --extract-stage 16:32:/data/stage1-checkpoint
#   ./setup_machine.sh --port 9195 --password '...' --name akun5 \
#       --extract-stage 0:16:/data/stage0-checkpoint:globals
#
# See --help for the full stage list and extract options.
#
# Qwen3.5's checkpoint is 78.8GB across 39 shards with no clean per-layer
# partition, so downloading it in full first would waste both bandwidth and
# disk headroom for no benefit. Instead the "extract" stage does a
# selective `hf download --include <only the shards this layer range
# touches>` (~53-57GB per 16-layer stage, measured real) folded together
# with extraction - see download_and_extract_qwen35_stage.py.
#
# DISK-SAFETY LESSON THIS SCRIPT ENFORCES BY DEFAULT (see
# memory/feedback_kaggle_working_placement.md - this exact mistake crashed
# two real machines this session): `df -h /` free space on these
# Kaggle-hosted machines does NOT reliably reflect real quota. After
# --extract-stage, the downloaded checkpoint shards are deleted unless
# --keep-full-checkpoint is passed explicitly.

set -euo pipefail

# ---- config ----
# LOCAL_ROOT is this repo's own root (the checkout containing this script
# at ops/setup_machine.sh) - auto-detected so this works regardless of
# clone location, not just the Kaggle convention. REMOTE_PROJECT_ROOT is
# where the SAME repo should live on the far side of each SSH connection;
# defaults to the Kaggle convention (this project's primary target
# environment - every path in pp_tests/launch/*.sh assumes it) but can be
# overridden if you're deploying somewhere else.
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/kaggle/working/vllm}"
SO_FILES=(
  "vllm/_moe_C_stable_libtorch.abi3.so"
  "vllm/third_party/deep_gemm/_C.cpython-310-x86_64-linux-gnu.so"
  "vllm/third_party/deep_gemm/_C.cpython-311-x86_64-linux-gnu.so"
  "vllm/third_party/deep_gemm/_C.cpython-312-x86_64-linux-gnu.so"
  "vllm/third_party/deep_gemm/_C.cpython-313-x86_64-linux-gnu.so"
  "vllm/third_party/deep_gemm/_C.cpython-314-x86_64-linux-gnu.so"
  "vllm/_flashmla_extension_C.abi3.so"
  "vllm/_flashmla_C.abi3.so"
  "vllm/cumem_allocator.abi3.so"
  "vllm/fs_io_C.abi3.so"
  "vllm/spinloop.abi3.so"
  "vllm/_qutlass_C.abi3.so"
  "vllm/_C_stable_libtorch.abi3.so"
  "vllm/_flashkda_C.abi3.so"
  "vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so"
  "vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so"
  "vllm/_rust_tool_parser.abi3.so"
)
TORCH_VERSION="2.13.0+cu130"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu130"
HUMMING_KERNELS_VERSION="0.1.10"
CHECKPOINT_DIR="/data/models/qwen3.5-122b-a10b-gptq"

# ---- args ----
PORT=""
PASSWORD=""
NAME="machine"
ONLY=""
EXTRACT_SPEC=""
KEEP_FULL_CHECKPOINT=0

usage() {
  cat <<EOF
Usage: $0 --port PORT --password PASS [--name NAME] [options]

Options:
  --port PORT                SSH port (required, e.g. 9195)
  --password PASS            SSH password (required)
  --name NAME                label for log output (default: machine)
  --only STAGE[,STAGE...]    run only these stages (code,torch,so,vllm,humming,extract,verify)
  --extract-stage S:E:OUT[:globals]
                              selectively download + extract layers [S,E) to OUT
                              (48 total hidden layers; append :globals for
                              embed_tokens/norm/lm_head - first/last stage only)
  --keep-full-checkpoint      do not delete the downloaded checkpoint shards at
                              $CHECKPOINT_DIR after --extract-stage. DANGEROUS if
                              anything else large is already on disk - see
                              memory/feedback_kaggle_working_placement.md.
  -h, --help                  show this help

Default stages (no --only): code torch vllm humming verify
("so" - manual .so copy from this machine - is opt-in only, see --only;
VLLM_USE_PRECOMPILED=1 already auto-downloads matching .so files)
(extract only runs if --extract-stage is given)
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --password) PASSWORD="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --extract-stage) EXTRACT_SPEC="$2"; shift 2 ;;
    --keep-full-checkpoint) KEEP_FULL_CHECKPOINT=1; shift ;;
    -h|--help) usage ;;
    *) echo "unknown arg: $1" >&2; usage ;;
  esac
done

[[ -n "$PORT" && -n "$PASSWORD" ]] || usage

# "so" (manual .so copy from this local machine) is NOT in the default
# set - confirmed on a real fresh machine that VLLM_USE_PRECOMPILED=1
# already auto-downloads matching precompiled .so files over the network
# keyed by git commit, making the manual copy redundant in the normal
# case. Kept available via --only ...,so,... as a fallback for a machine
# whose network can't reach vLLM's wheel server.
DEFAULT_STAGES="code,torch,vllm,humming,verify"
if [[ -n "$ONLY" ]]; then
  STAGE_CSV="$ONLY"
else
  STAGE_CSV="$DEFAULT_STAGES"
  [[ -n "$EXTRACT_SPEC" ]] && STAGE_CSV="$STAGE_CSV,extract"
fi
STAGE_CSV=",$STAGE_CSV,"

log() { echo "[$NAME] $*" >&2; }
has_stage() { [[ "$STAGE_CSV" == *",$1,"* ]]; }

# Every remote command is a single string, cd'd into the project root
# first - this project lost real time earlier this session to forgetting
# that step; never repeat it.
rssh() {
  sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p "$PORT" root@127.0.0.1 \
    "cd $REMOTE_PROJECT_ROOT 2>/dev/null; $1"
}

# ---- stages ----

stage_code() {
  log "syncing repo (vllm package incl. .git, humming_fix, udp_holepunch, ops, pp_tests, scripts)..."
  sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -p "$PORT" root@127.0.0.1 \
    "mkdir -p $REMOTE_PROJECT_ROOT"
  sshpass -p "$PASSWORD" rsync -az -e "ssh -o StrictHostKeyChecking=no -p $PORT" \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \
    "$LOCAL_ROOT/." "root@127.0.0.1:$REMOTE_PROJECT_ROOT/"
  log "code sync done"
}

stage_torch() {
  local current
  current=$(rssh "python3 -c 'import torch; print(torch.__version__)' 2>/dev/null" || true)
  if [[ "$current" == "$TORCH_VERSION" ]]; then
    log "torch already at $TORCH_VERSION, skipping"
    return
  fi
  log "installing torch==$TORCH_VERSION (was: ${current:-none})..."
  rssh "pip install --index-url $TORCH_INDEX_URL torch==$TORCH_VERSION 2>&1 | tail -20"
}

stage_so() {
  log "copying ${#SO_FILES[@]} precompiled .so files..."
  local f
  # SO_FILES entries are paths relative to this repo's root (e.g.
  # "vllm/_C_stable_libtorch.abi3.so" = repo_root/vllm/_C_stable_libtorch.abi3.so,
  # i.e. the inner vllm package dir) - $LOCAL_ROOT/$REMOTE_PROJECT_ROOT
  # already point at the repo root, so no extra "vllm/" prefix needed here.
  # Verify with `find "$LOCAL_ROOT" -name '*.so'` after editing this if it
  # ever looks wrong again.
  for f in "${SO_FILES[@]}"; do
    sshpass -p "$PASSWORD" scp -o StrictHostKeyChecking=no -P "$PORT" \
      "$LOCAL_ROOT/$f" "root@127.0.0.1:$REMOTE_PROJECT_ROOT/$f" \
      || log "WARNING: failed to copy $f (missing locally? build it first)"
  done
  log ".so copy done"
}

stage_vllm() {
  rssh "pip install setuptools_rust setuptools_scm 2>&1 | tail -5"
  log "installing vllm (VLLM_USE_PRECOMPILED=1, .so-copy + dummy-wheel path)..."
  # setup.py's get_vllm_version()/determine_wheel_url() try `git fetch
  # https://github.com/vllm-project/vllm main` + merge-base to find an
  # ancestor commit with a published precompiled wheel when
  # VLLM_USE_PRECOMPILED=1 is set - real, reproducible, NON-transient
  # failure found this session: this project's vllm/.git is a
  # "reconstructed" 2-commit repo (see its own log), not a real clone of
  # vllm-project/vllm, so `git merge-base` against a freshly-fetched
  # upstream ref ALWAYS fails (no common ancestor exists). This makes
  # setup.py fall back to "nightly", and the nightly wheel repo has been
  # seen to have NO x86_64 build published (only aarch64) - retrying does
  # not fix this, every attempt hits the same wall. Confirmed on 3
  # separate real machines this session - always take the manual path,
  # don't waste time on the auto-fetch attempt first.
  #
  # The fix: copy this local machine's own known-good .so files (already
  # built once here) to the remote, then point
  # VLLM_PRECOMPILED_WHEEL_LOCATION at a real-but-empty local zip (a wheel
  # is just a zip) - setup.py's determine_wheel_url() takes the
  # "use existing wheel at <path>" branch and skips network entirely,
  # extracts zero members from the empty archive (harmless - our .so
  # files are already manually placed at their final locations), and
  # metadata generation completes normally. This same dummy-wheel env var
  # MUST be set on every subsequent `pip install` call in this function,
  # not just the first - real bug hit running this for real: omitting it
  # on the second (full-deps) install call let that call fall through to
  # the same broken auto-fetch path, which failed non-fatally enough that
  # `pip install -e .` still exited 0 (kernels were already importable
  # from the first pass) while leaving `transformers`/`cbor2`/etc at their
  # base-image versions instead of the versions vllm actually needs -
  # `pip show vllm` came back "not found" and `vllm serve` crashed on
  # `ModuleNotFoundError: No module named 'cbor2'` at first real use, long
  # after this stage had already reported success.
  stage_so
  rssh "python3 -c \"import zipfile; zipfile.ZipFile('/tmp/dummy_vllm_wheel.whl', 'w').close()\""
  rssh "timeout 480 env VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION=/tmp/dummy_vllm_wheel.whl pip install -e . --no-build-isolation --no-deps 2>&1 | tail -30"
  rssh "timeout 900 env VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION=/tmp/dummy_vllm_wheel.whl pip install -e . --no-build-isolation 2>&1 | tail -40"
  local ok
  ok=$(rssh "python3 -c 'import vllm._C_stable_libtorch; print(\"COMPILED_KERNELS_OK\")' 2>&1" || true)
  if [[ "$ok" != *COMPILED_KERNELS_OK* ]]; then
    log "ERROR: vllm compiled-kernel import failed:"
    log "$ok"
    return 1
  fi
  # Real bug this check would have caught immediately instead of surfacing
  # at first `vllm serve` use: `pip install -e .` can exit 0 while the
  # vllm package itself isn't actually registered/importable outside its
  # own directory (dependency resolution silently incomplete) - verify the
  # FULL entrypoint import chain, not just the compiled-kernel .so, and
  # confirm pip actually sees the package as installed.
  local full_import
  full_import=$(rssh "python3 -c 'import vllm.entrypoints.cli.main; print(\"FULL_IMPORT_OK\")' 2>&1" || true)
  if [[ "$full_import" != *FULL_IMPORT_OK* ]]; then
    log "ERROR: vllm full entrypoint import failed (likely a missing/mismatched Python dependency, not the compiled kernels) - re-running full install..."
    rssh "timeout 900 env VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION=/tmp/dummy_vllm_wheel.whl pip install -e . --no-build-isolation 2>&1 | tail -40"
    full_import=$(rssh "python3 -c 'import vllm.entrypoints.cli.main; print(\"FULL_IMPORT_OK\")' 2>&1" || true)
    if [[ "$full_import" != *FULL_IMPORT_OK* ]]; then
      log "ERROR: vllm full entrypoint import still failing:"
      log "$full_import"
      return 1
    fi
  fi
  log "vllm install verified (compiled kernels import OK)"
}

stage_humming() {
  local ver
  ver=$(rssh "pip show humming-kernels 2>/dev/null | grep Version" || true)
  if [[ "$ver" == *"$HUMMING_KERNELS_VERSION"* ]]; then
    log "humming-kernels already at $HUMMING_KERNELS_VERSION, skipping"
    return
  fi
  log "installing humming-kernels==$HUMMING_KERNELS_VERSION..."
  rssh "pip install humming-kernels==$HUMMING_KERNELS_VERSION 2>&1 | tail -10"
}

stage_extract() {
  [[ -n "$EXTRACT_SPEC" ]] || { log "no --extract-stage given, skipping"; return; }
  local start end out globals flag keep_flag
  IFS=':' read -r start end out globals <<< "$EXTRACT_SPEC"
  flag=""
  [[ "$globals" == "globals" ]] && flag="--include-globals"
  keep_flag=""
  [[ "$KEEP_FULL_CHECKPOINT" -eq 1 ]] && keep_flag="--keep-full-checkpoint"

  log "selectively downloading + extracting layers [$start,$end) -> $out $flag ..."
  rssh "cd $REMOTE_PROJECT_ROOT/humming_fix/single_layer_probe && python3 download_and_extract_qwen35_stage.py --start $start --end $end --out $out --checkpoint-dir $CHECKPOINT_DIR $flag $keep_flag 2>&1 | tail -40"
}

stage_verify() {
  log "final verification..."
  rssh "python3 -c 'import torch; print(\"torch:\", torch.__version__); print(\"cuda:\", torch.cuda.is_available()); print(\"gpus:\", torch.cuda.device_count())'"
  rssh "python3 -c 'import vllm._C_stable_libtorch; print(\"vllm compiled kernels: OK\")'"
  rssh "pip show humming-kernels 2>&1 | grep Version"
  rssh "nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader"
  rssh "df -h / | tail -1"
  log "verification done"
}

# ---- run (fixed order regardless of --only membership) ----
log "starting setup on port $PORT: stages=[$STAGE_CSV]"
for s in code torch so vllm humming extract verify; do
  if has_stage "$s"; then
    log "=== stage: $s ==="
    "stage_$s"
  fi
done
log "all requested stages complete"
