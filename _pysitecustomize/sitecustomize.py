"""Auto-imported by every Python interpreter start-up when this
directory (`_pysitecustomize/`, a dedicated directory containing only
this file - deliberately NOT the project root, see below) is on
PYTHONPATH (standard `site` module behavior - see
https://docs.python.org/3/library/site.html#module-sitecustomize).

Exists so the real SM75 `humming-kernels` runtime patches
(`humming_fix/patch.py`, see that module's docstring for what they fix)
AND the Qwen3.5 MTP + synthetic-PP compatibility patch
(`vllm/transport/qwen35_mtp_pp_fix.py`, see that module's docstring)
apply automatically to `vllm serve` itself - `scripts/launch_pp_stage.py`
`os.execvpe`s directly into vLLM's own CLI entrypoint, so there is no
other point in that process where this project's own code runs first to
apply the patch by hand (every other script in this project imports
`humming_fix.patch` explicitly instead; this file exists only to cover
that one CLI-subprocess case).

Loads patch.py by direct file path, NOT via a normal `import
humming_fix.patch` + sys.path entry - this project's root directory also
contains a directory literally named `vllm` (the vLLM checkout), so
putting the project root itself on sys.path/PYTHONPATH creates a
namespace-package collision with the real editable-installed `vllm`
package (hit this for real: `ImportError: cannot import name
'SamplingParams' from 'vllm' (unknown location)` - "unknown location" is
the signature of a broken merged namespace package). That's why this
file lives in its own dedicated `_pysitecustomize/` directory instead of
the project root - PYTHONPATH only ever points at this directory, which
has no `vllm`-named sibling to collide with. patch.py itself only
imports from the real `humming` package + stdlib (no relative imports),
so direct file-path loading (below) is safe on top of that.
"""
import importlib.util
import sys
from pathlib import Path

_patch_path = Path(__file__).resolve().parent.parent / "humming_fix" / "patch.py"
if _patch_path.exists():
    _spec = importlib.util.spec_from_file_location("humming_fix_patch", _patch_path)
    _module = importlib.util.module_from_spec(_spec)
    sys.modules["humming_fix_patch"] = _module
    _spec.loader.exec_module(_module)

# Real vllm subpackage (not this project's own root dir), safe to import
# normally - only meaningful/active when --speculative-config builds an
# MTP drafter, harmless no-op import otherwise.
try:
    import vllm.transport.qwen35_mtp_pp_fix  # noqa: F401
except ImportError:
    pass
