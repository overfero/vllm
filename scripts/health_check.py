#!/usr/bin/env python3
"""Post-launch health check for one pipeline stage - run this AFTER
`launch_pp_stage.py` has been started on a machine, to verify it came up
correctly before considering that machine "ready."

Every stage (per launch_pp_stage.py's module docstring) runs a real vLLM
HTTP server bound to 127.0.0.1 even when --serve wasn't passed, so this
script's process-liveness/model-loaded checks work identically on every
machine; the completion-request check only applies to the stage that was
launched with --serve (the one exposing 0.0.0.0 and meant to take real
client traffic).

What this actually verifies, and what it can't:

- Process alive + engine responsive: real, via vLLM's own `/health`
  endpoint (`vllm/entrypoints/serve/instrumentator/health.py` - reused
  unmodified).
- Checkpoint + tokenizer loaded, model name matches: real, via
  `/v1/models`.
- First generated token: real, via an actual `/v1/completions` request -
  only meaningful on the --serve stage, and only actually exercises the
  full pipeline if the Executor/scheduler_output gap documented in
  launch_pp_stage.py and README_RUN_GPTOSS_CLUSTER.md has been closed -
  otherwise this call will hang or error, which is itself a correct and
  useful diagnostic (see the "Known gap" section).
- TP initialized / PP initialized / transport connected / neighbor
  connected / KV cache initialized: NOT independently queryable through
  any existing vLLM HTTP endpoint. The closest real signal is grepping
  this stage's own stdout/stderr log for the specific lines
  TransportPPWorker and vLLM's own bootstrap already print (see
  `--log-file` below) - this script does that if given a log path, but
  it is line-matching on log text, not a live state query, and will
  under-report if the process's logs were redirected somewhere else.

Usage:

    python3 scripts/health_check.py --host 127.0.0.1 --port 8080 \\
        --model-path /data/models/gpt-oss-120b-gptq \\
        [--completion] [--log-file /var/log/gptoss/machineC.log] \\
        [--timeout 600]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

EXPECTED_LOG_CHECKPOINTS = [
    ("local TP/DP/PP groups formed (real torch.distributed)", "rank 0 in world size"),
    ("hole punch to neighbor succeeded", "Hole punch success."),
    ("transport PP group installed", "transport PP group installed and connected"),
    ("model weights loaded", "Loading model weights took"),
]


def _get(url: str, timeout: float) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def check_health(base_url: str, timeout: float, retries: int, retry_interval: float) -> bool:
    for attempt in range(1, retries + 1):
        try:
            status, _ = _get(f"{base_url}/health", timeout=timeout)
            if status == 200:
                print(f"[health] /health: OK (attempt {attempt}/{retries})")
                return True
            print(f"[health] /health: HTTP {status} (attempt {attempt}/{retries})")
        except Exception as exc:  # noqa: BLE001
            print(f"[health] /health: unreachable - {type(exc).__name__}: {exc} "
                  f"(attempt {attempt}/{retries})")
        if attempt < retries:
            time.sleep(retry_interval)
    return False


def check_models(base_url: str, expected_model_path: str | None, timeout: float) -> bool:
    try:
        status, body = _get(f"{base_url}/v1/models", timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        print(f"[health] /v1/models: unreachable - {type(exc).__name__}: {exc}")
        return False
    if status != 200:
        print(f"[health] /v1/models: HTTP {status}: {body}")
        return False
    try:
        data = json.loads(body)
        model_ids = [m.get("id") for m in data.get("data", [])]
    except Exception as exc:  # noqa: BLE001
        print(f"[health] /v1/models: could not parse response - {exc}")
        return False
    print(f"[health] /v1/models: OK, served model id(s): {model_ids}")
    if expected_model_path and expected_model_path not in model_ids:
        print(f"[health] WARNING: expected model path {expected_model_path!r} "
              f"not among served ids {model_ids} - check --model matches "
              f"across all 3 machines' launch commands")
    return True


def check_completion(base_url: str, timeout: float) -> bool:
    payload = json.dumps({"prompt": "Hello", "max_tokens": 1}).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8", "replace"))
    except Exception as exc:  # noqa: BLE001
        print(f"[health] /v1/completions: FAILED - {type(exc).__name__}: {exc}")
        print("[health] If this hung until --timeout: see launch_pp_stage.py's "
              "'Known gap' docstring section and README_RUN_GPTOSS_CLUSTER.md - "
              "this is the expected symptom of the unresolved Executor/"
              "scheduler_output cross-machine RPC gap, not a transport failure.")
        return False
    try:
        text = body["choices"][0]["text"]
    except Exception:  # noqa: BLE001
        print(f"[health] /v1/completions: unexpected response shape: {body}")
        return False
    print(f"[health] /v1/completions: OK, first generated token/text: {text!r}")
    return True


def check_log_checkpoints(log_file: str) -> None:
    try:
        with open(log_file, errors="replace") as f:
            content = f.read()
    except OSError as exc:
        print(f"[health] could not read --log-file {log_file}: {exc}")
        return
    for label, needle in EXPECTED_LOG_CHECKPOINTS:
        found = needle in content
        print(f"[health] log checkpoint - {label}: {'FOUND' if found else 'missing'}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--model-path", default=None)
    p.add_argument("--completion", action="store_true", help="also send a real /v1/completions request")
    p.add_argument("--log-file", default=None, help="grep this stage's log for expected bootstrap checkpoints")
    p.add_argument("--timeout", type=float, default=600.0, help="per-request timeout, seconds (model load can be slow)")
    p.add_argument("--retries", type=int, default=30)
    p.add_argument("--retry-interval", type=float, default=10.0)
    args = p.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    print(f"[health] checking {base_url} ...")

    ok = check_health(base_url, args.timeout, args.retries, args.retry_interval)
    if not ok:
        print("[health] ABORT: /health never returned 200 - engine process is "
              "not up, crashed during load, or the port/host is wrong.",
              file=sys.stderr)
        if args.log_file:
            check_log_checkpoints(args.log_file)
        return 1

    ok = check_models(base_url, args.model_path, args.timeout) and ok
    if args.completion:
        ok = check_completion(base_url, args.timeout) and ok
    if args.log_file:
        check_log_checkpoints(args.log_file)

    print("\nHEALTH CHECK " + ("OK" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
