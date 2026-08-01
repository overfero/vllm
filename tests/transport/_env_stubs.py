"""Sandbox-only import shims - NOT vLLM behavior, NOT part of this
project's transport/pipeline work. These exist solely because this
specific sandbox's preinstalled dependency/build state predates or departs
from what vLLM's current main branch expects, and block importing vLLM at
all (or block real GPU execution) otherwise. A properly-provisioned
deployment machine (matching vLLM's pinned requirements, built with
`pip install`, with compiled CUDA kernels present) should not need any of
this - see README_GPTOSS_120B_UDP.md ("Phase 5") for the full diagnosis of
each item below, including exactly what would make it unnecessary.

Import this module FIRST, before anything that transitively imports vllm,
in any script that needs the real (not `_pipeline_shim.py`-bypassed)
`vllm.config`/`vllm.distributed`/`vllm` engine import chain.

What each shim is for, in the order they must apply:

1. Force CPU platform selection (must happen before the FIRST import of
   any `vllm` submodule, since platform auto-detection is a package-
   import-time side effect - `import vllm.X` for any X triggers it).
   This checkout was never `pip install`-ed, so it has no compiled
   `vllm._C_stable_libtorch` / `vllm._moe_C_stable_libtorch` custom CUDA
   kernel extensions. Real GPUs ARE present and pynvml correctly detects
   them, which makes vllm.platforms auto-select CudaPlatform - which then
   crashes importing the missing compiled extension. Monkeypatching
   pynvml's device count to 0 (via the raw `pynvml` package, imported
   directly - NOT via vLLM's own wrapper, which would itself trigger the
   same crash) makes vLLM's own platform auto-detection correctly fall
   through to CpuPlatform instead, which has no compiled-extension
   requirement. This means real GPU compute is NOT exercised by anything
   that imports this module - only the engine/bootstrap/distributed
   layers are, on CPU. That is a real, working proof of the architecture;
   it is not a proof of GPU inference performance, which remains blocked
   on a full CUDA kernel build (nvcc is present in this sandbox, but a
   full vLLM kernel build was judged too slow/risky to attempt blind
   within this project's time budget - see the README for the full
   assessment).

2. `xgrammar` -> MagicMock: this sandbox's preinstalled xgrammar==0.2.4
   needs a compiled tvm_ffi/xgrammar_bindings native library that isn't
   available here (its own `tvm_ffi` PyPI dependency is yanked/missing).
   vllm.v1.structured_output imports xgrammar unconditionally at module
   load regardless of whether guided/grammar-constrained decoding is ever
   used, so without this, nothing in vllm.v1 (engine, request, scheduler)
   can be imported. Guided decoding will not work with this stub - this
   project's tests never request it.

3. `torch.float4_e2m1fn_x2`: this sandbox's torch==2.6.0+cu124 predates
   the float4_e2m1fn_x2 dtype (added in a later torch release) that
   vllm/ir/tolerances.py references unconditionally at import time (as a
   dict key). GPT-OSS's native MXFP4 quantization genuinely needs this
   dtype to *work* - this stub only lets modules that reference it as a
   dict key import without crashing; it does not make FP4 tensors/kernels
   functional. Real FP4 quantized inference is not expected to work in
   this environment regardless of this stub.
"""
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

# vLLM vendors its own copy of pynvml at vllm/third_party/pynvml.py (see
# vllm/utils/import_utils.py:import_pynvml's docstring for why) - patching
# the standalone top-level `pynvml` PyPI package (if any) has no effect on
# it. It must be loaded and patched *before* `import vllm` ever runs (even
# transitively), because `import vllm` executes vllm/__init__.py, which
# eagerly triggers platform auto-detection as a side effect - so there is
# no point after which "import vllm normally, then patch" would work.
# Loading the file directly via importlib (bypassing the `vllm` package
# namespace, which doesn't exist in sys.modules yet) and pre-registering it
# under its real dotted name means vLLM's own internal
# `import vllm.third_party.pynvml as pynvml` (run later, from inside
# vllm/__init__.py's own import chain) finds it already cached and reuses
# our patched copy instead of re-executing the file.
_pynvml_path = Path(__file__).resolve().parents[2] / "vllm" / "third_party" / "pynvml.py"
_spec = importlib.util.spec_from_file_location("vllm.third_party.pynvml", _pynvml_path)
_pynvml_module = importlib.util.module_from_spec(_spec)

sys.modules.setdefault("vllm.third_party", types.ModuleType("vllm.third_party"))
sys.modules["vllm.third_party.pynvml"] = _pynvml_module  # must be registered before exec - the
sys.modules["vllm.third_party"].pynvml = _pynvml_module  # module's own body does sys.modules[__name__]

_spec.loader.exec_module(_pynvml_module)
_pynvml_module.nvmlDeviceGetCount = lambda: 0

if "xgrammar" not in sys.modules:
    sys.modules["xgrammar"] = MagicMock()

import torch as _torch

if not hasattr(_torch, "float4_e2m1fn_x2"):
    _torch.float4_e2m1fn_x2 = "float4_e2m1fn_x2 (stub - not a real torch dtype in this torch version)"
