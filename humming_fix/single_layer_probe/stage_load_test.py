"""Generalized real vLLM LLM(...) load+generate test for an arbitrary
stage/layer-count checkpoint. Used for incremental validation
(2 -> 4 -> 8 -> 12 layers) before attempting real cross-machine PP.

Usage:
    python3 stage_load_test.py --model /stage-test-2layer --tp 1
    python3 stage_load_test.py --model /stage-test-2layer --tp 2
"""
import argparse
import os

import humming_fix.patch  # noqa: E402  (both SM75 fixes, before any HummingKernel use)

import torch  # noqa: E402

torch.zeros(1, device="cuda")
torch.cuda.synchronize()

from vllm import LLM, SamplingParams  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=4)
    args = ap.parse_args()

    print(f"=== Constructing LLM(model={args.model}, tp={args.tp}) ===", flush=True)
    llm = LLM(
        model=args.model,
        quantization="gptq",
        dtype="float16",
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=0.5,
        max_model_len=128,
        enforce_eager=True,
        trust_remote_code=True,
    )
    print("=== LLM constructed OK - real weights loaded, no crash ===", flush=True)

    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    outputs = llm.generate(["Hello, my name is"], sp)
    for out in outputs:
        print("GENERATED TOKEN IDS:", out.outputs[0].token_ids)
        print("GENERATED TEXT:", repr(out.outputs[0].text))

    print(f"\n=== SUCCESS: model={args.model} tp={args.tp} - real forward pass completed ===")


if __name__ == "__main__":
    main()
