"""Phase 2B exit criteria from `vllm/README_ARCHITECTURE_DECISION.md`:

1. Litmus test: `stage_runner.py` (the only file that runs the actual
   pipeline logic) never references vLLM/SGLang-specific vocabulary.
   Enforced here by grep, not just a one-time manual read - a future edit
   that quietly reintroduces framework coupling fails this test.
2. Correctness: the pipeline's output matches a directly-computed,
   non-pipelined ground truth - proving the transport actually moved the
   right data, not just that the processes didn't crash.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_STAGE_RUNNER = _DIR / "stage_runner.py"
_RUN_DEMO = _DIR / "run_demo.py"

_FORBIDDEN_TERMS = [
    "GroupCoordinator",
    "Worker",
    "PP rank",
    "pp_rank",
    "send_tensor_dict",
    "vllm",
    "sglang",
]


def test_litmus_no_framework_vocabulary_in_stage_runner() -> None:
    source = _STAGE_RUNNER.read_text()
    hits = [term for term in _FORBIDDEN_TERMS if term.lower() in source.lower()]
    assert not hits, (
        f"stage_runner.py references framework-specific vocabulary {hits} - "
        "the runtime is supposed to be usable with zero knowledge of any "
        "particular inference framework (see README_ARCHITECTURE_DECISION.md "
        "Part 5's litmus test)"
    )


def test_pipeline_output_matches_ground_truth() -> None:
    result = subprocess.run(
        [sys.executable, str(_RUN_DEMO)],
        cwd=_DIR,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"run_demo.py failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    assert "OK: pipeline output matches non-pipelined ground truth" in result.stdout
