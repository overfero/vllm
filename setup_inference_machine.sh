#!/bin/bash
# One-time setup for a quic-vllm (inference) machine. Idempotent - safe
# to re-run. Real recipe learned recovering from a full environment
# reset - every step here turned out to matter for real, not a guess.
#
# Builds vllm in its OWN isolated venv (/vllm_build_venv, under root -
# not /kaggle/working, which is comparatively small and where this
# venv's real ~7GB (mostly torch's own bundled CUDA libraries) noticeably
# ate into free space during testing; this project's own established
# convention is large downloads go under / - see README/CLAUDE.md)
# rather than the main environment - two real reasons:
#  1. vllm's runtime requirement (requirements/cuda.txt: torch==2.13.0)
#     would otherwise silently upgrade whatever torch the main
#     environment already has (this project's own quic-train/quic-rl
#     work uses a different torch version there) - isolating it avoids
#     that entirely, matching how pip's OWN build isolation already
#     protects the main env during a normal `pip install -e .` (that
#     mechanism just doesn't let us trim the target list - see below).
#  2. It lets this script pass `--no-build-isolation` + a trimmed
#     CMake target list (VLLM_BUILD_EXTENSIONS, a real addition to
#     setup.py's build_extensions() - see that patch's own comment)
#     safely, without needing pip's build isolation to also protect
#     the main env - this venv already does that job.
#
# Real, hard-learned specifics:
# - TORCH_CUDA_ARCH_LIST must be EXPORTED as its own statement, not
#   prefixed inline (`VAR=val pip install ...`) - the inline form was
#   found NOT to propagate into pip's actual build subprocess in this
#   harness (confirmed via /proc/PID/environ).
# - protobuf-compiler/libprotobuf-dev are needed for the Rust build's
#   tonic/prost crate - a real, easy-to-miss apt dependency.
# - `ensurepip` is missing from this environment's Python - venvs need
#   pip bootstrapped via get-pip.py instead of the usual `python -m venv`
#   default. `python3-venv` itself may also need installing first.
# - Trimmed build (default): only compiles kernels this hardware can
#   actually use (_C_stable_libtorch, _moe_C_stable_libtorch,
#   _vllm_fa2_C, cumem_allocator, spinloop, fs_io_C, triton_kernels) -
#   skips every Hopper/Blackwell-only kernel (_vllm_fa3_C,
#   _vllm_fa4_cutedsl_C, _flashkda_C, _deep_gemm_C, _qutlass_C,
#   fmha_sm100, tml_fa4) this project's own vLLM/vllm-flash-attn
#   CMakeLists.txt don't reliably skip on their own even with
#   TORCH_CUDA_ARCH_LIST set correctly (confirmed via a real nvcc
#   -gencode=sm_90 inspection - a real, unresolved upstream gap, not
#   guessed). Cuts real compile work roughly to a third (~112 of 340
#   object files, measured directly via ninja's own dependency graph).
#   Set VLLM_FULL_BUILD=1 to build everything instead (e.g. for
#   different/newer hardware that actually needs those kernels).
#
# USAGE NOTE: the built vllm lives in THIS venv, not system python3 -
# any script that needs it (scripts/stage_server.py, launch_pp_stage.py,
# a real `vllm serve`) must be run with `$VLLM_BUILD_VENV/bin/python3`
# (default /vllm_build_venv/bin/python3), not plain python3.
#
# The venv is location-independent once built - `bin/python3` is a real
# symlink (works after a move), but `bin/pip` bakes an absolute shebang
# path at creation time and breaks if moved (confirmed for real by moving
# this venv from under the repo to /). This script therefore always
# invokes pip via `python3 -m pip`, never the `pip`/`pip3` scripts
# directly, so a relocated venv keeps working with zero reinstall.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5}"
VENV_DIR="${VLLM_BUILD_VENV:-/vllm_build_venv}"
TRIMMED_TARGETS="_C_stable_libtorch,_moe_C_stable_libtorch,_vllm_fa2_C,cumem_allocator,spinloop,fs_io_C,triton_kernels"

echo "=== building for TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST ==="

echo "=== [1/5] apt deps (protobuf for the Rust build, venv support) ==="
apt-get update -qq
apt-get install -y -qq protobuf-compiler libprotobuf-dev python3-venv

echo "=== [2/5] isolated build venv ==="
if [ ! -x "$VENV_DIR/bin/python3" ]; then
    python3 -m venv --without-pip "$VENV_DIR"
    curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    "$VENV_DIR/bin/python3" /tmp/get-pip.py -q
fi
"$VENV_DIR/bin/python3" -m pip install -q cmake ninja "packaging>=24.2" "setuptools>=77.0.3,<81.0.0" \
    "setuptools-scm>=8.0" "setuptools-rust>=1.9.0" wheel jinja2 torch==2.13.0

echo "=== [3/5] vllm Python package + CUDA/C++ extensions ==="
if [ "${VLLM_FULL_BUILD:-0}" = "1" ]; then
    echo "    VLLM_FULL_BUILD=1 - building every kernel (slow, 30-90+ min)"
    "$VENV_DIR/bin/python3" -m pip install --no-build-isolation -e . -q
else
    echo "    trimmed build - only T4-relevant kernels (~1/3 the work)"
    VLLM_BUILD_EXTENSIONS="$TRIMMED_TARGETS" "$VENV_DIR/bin/python3" -m pip install --no-build-isolation -e . -q
fi

echo "=== [4/5] Rust extensions (QUIC/UDP transport) ==="
if ! command -v cargo &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
    source "$HOME/.cargo/env"
fi
source "$HOME/.cargo/env" 2>/dev/null || true
"$VENV_DIR/bin/python3" -m pip install -q setuptools_rust  # tools/build_rust.py needs it in-venv, not just the isolated overlay
# build_rust.sh hardcodes a bare `python3` call (no env var override) -
# prepend the venv's bin so THAT resolves first, matching the Python
# vllm itself is actually installed into.
PATH="$VENV_DIR/bin:$PATH" ./build_rust.sh

echo "=== [5/5] verify (from a neutral cwd - /kaggle/working IS the repo's own"
echo "    parent, so testing from inside it can shadow the real install with"
echo "    an implicit namespace package - a real mistake made building this"
echo "    script, not a hypothetical) ==="
(cd /tmp && "$VENV_DIR/bin/python3" -c "
import vllm
print('vllm OK:', vllm.__file__)
import vllm._C_stable_libtorch
import vllm._moe_C_stable_libtorch
import vllm.vllm_flash_attn._vllm_fa2_C
print('all core extensions import OK')
")

echo "=== done ==="
echo "Use $VENV_DIR/bin/python3 to run vllm (scripts/stage_server.py, launch_pp_stage.py, etc) -"
echo "NOT the system python3, which never gets vllm installed by this script."
