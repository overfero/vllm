#!/bin/bash
# One-time setup for a quic-vllm (inference) machine. Idempotent - safe
# to re-run. Real recipe learned recovering from a full environment
# reset - every step here turned out to matter for real, not a guess:
#
# - TORCH_CUDA_ARCH_LIST must be EXPORTED as its own statement, not
#   prefixed inline (`VAR=val pip install ...`) - the inline form was
#   found NOT to propagate into pip's actual build subprocess in this
#   harness (confirmed via /proc/PID/environ), silently building for
#   every default architecture (including Hopper/Blackwell-only kernels
#   this hardware can never use) instead of just this GPU's.
# - protobuf-compiler/libprotobuf-dev are needed for the Rust build's
#   tonic/prost crate (google/protobuf/struct.proto) - a real, easy-to-
#   miss apt dependency, not bundled with protobuf-compiler alone.
# - Set this to the ACTUAL target GPU's compute capability - 7.5 for
#   Tesla T4 (Turing). Override via env before running this script for
#   different hardware, e.g. `TORCH_CUDA_ARCH_LIST=8.0 ./setup_inference_machine.sh`.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5}"
echo "=== building for TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST ==="

echo "=== [1/3] apt deps for the Rust build (protobuf) ==="
apt-get update -qq
apt-get install -y -qq protobuf-compiler libprotobuf-dev

echo "=== [2/3] vllm Python package + CUDA/C++ extensions (the slow step - a genuine"
echo "    from-source compile, likely 30-60+ minutes; some vendored subprojects"
echo "    (vllm-flash-attn's Hopper-only FA3 kernels) don't fully respect the arch"
echo "    restriction and compile anyway - a known, real, unresolved gap, not"
echo "    something this script papers over) ==="
pip install -e . -q

echo "=== [3/3] Rust extensions (QUIC/UDP transport) ==="
./build_rust.sh

echo "=== verify vllm actually imports ==="
python3 -c "import vllm; print('vllm OK:', vllm.__file__)"

echo "=== done ==="
echo "Consider snapshotting the compiled .so's for next time:"
echo "  rm -rf ../.vllm_so_backup && mkdir -p ../.vllm_so_backup/vllm"
echo "  cp -r vllm/*.so vllm/vllm_flash_attn vllm/third_party ../.vllm_so_backup/vllm/ 2>/dev/null || true"
echo "  (only reusable on a machine with the SAME CUDA runtime major.minor -"
echo "   a backup built against a different CUDA version will fail to import"
echo "   with a real, confirmed 'libcudart.so.N: cannot open shared object file'"
echo "   error, not a guess - this bit us once already recovering this exact"
echo "   environment)."
