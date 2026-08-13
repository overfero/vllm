"""Quick idempotency check for setup_machine.sh: does a checkpoint
directory already contain the expected, complete safetensors index?
Deliberately simple (no deep tensor validation) - just enough to decide
"skip the 61GB re-download" safely. Exits 0 (and prints the size) if it
matches, exits 1 (silently, no traceback) otherwise - callers should
treat any non-zero exit as "not present / needs (re)download".

Usage:
    python3 verify_checkpoint.py /data/models/gpt-oss-120b-gptq 64862418823
"""
import json
import os
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: verify_checkpoint.py <dir> <expected_total_size>", file=sys.stderr)
        return 1
    checkpoint_dir, expected_size = sys.argv[1], int(sys.argv[2])
    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return 1
    try:
        with open(index_path) as f:
            idx = json.load(f)
        actual_size = idx["metadata"]["total_size"]
    except (json.JSONDecodeError, KeyError, OSError):
        return 1
    if actual_size != expected_size:
        return 1
    print(actual_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
