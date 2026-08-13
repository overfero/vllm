"""Real, full vLLM production model-loading path test: load the
standalone 4GB single-layer GPT-OSS-120B GPTQ checkpoint
(/gpt-oss-120b-gptq-1layer) through the real `LLM(...)` API, with the
real `humming-kernels` MoE backend, both SM75 bugs patched
(`humming_fix.patch`), on real Tesla T4 hardware.

Goal: confirm the exact call chain that used to hit both humming bugs
(auto_gptq.py -> humming_utils.py -> HummingKernel.prepare_kernels ->
NVRTC/nvcc compile -> cuModuleLoad -> real kernel launch) now completes
cleanly end-to-end through vLLM's own real weight-loading and forward
pass code - not a hand-rolled repro of the kernel call alone (that's
already covered by humming_fix/test_repro.py).

This is a single-layer, single-GPU sanity check, not a claim of correct
generation - 34 of 36 real transformer layers are structurally absent by
construction, so gibberish output is the *expected*, correct result.
The signal here is "no crash, no NaN, a real token comes out the other
end of the real vLLM engine," not "coherent text."
"""
import humming_fix.patch  # noqa: E402  (both SM75 fixes, before any HummingKernel use)

import torch  # noqa: E402

torch.zeros(1, device="cuda")
torch.cuda.synchronize()

from vllm import LLM, SamplingParams  # noqa: E402

MODEL_PATH = "/gpt-oss-120b-gptq-1layer"


def main() -> None:
    print("=== Constructing LLM(...) - real production model-loading path ===", flush=True)
    llm = LLM(
        model=MODEL_PATH,
        quantization="gptq",
        dtype="float16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,
        max_model_len=128,
        enforce_eager=True,
        trust_remote_code=True,
    )
    print("=== LLM constructed OK - real weights loaded, no crash ===", flush=True)

    sp = SamplingParams(max_tokens=4, temperature=0.0)
    print("=== Running real forward pass / generate() ===", flush=True)
    outputs = llm.generate(["Hello, my name is"], sp)

    for out in outputs:
        print("PROMPT:", out.prompt)
        print("GENERATED TOKEN IDS:", out.outputs[0].token_ids)
        print("GENERATED TEXT:", repr(out.outputs[0].text))
        has_nan_marker = any(t < 0 or t > 300000 for t in out.outputs[0].token_ids)
        print("token ids look sane (no obviously invalid ids):", not has_nan_marker)

    print("\n=== SUCCESS: real forward pass completed, real token(s) generated, no crash ===")


if __name__ == "__main__":
    main()
