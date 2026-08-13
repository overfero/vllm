# Single-layer engine-level load probe

Reconstructed and re-run for real on 2026-08-03, after the working
directory that originally produced this (per `docs/ARCHITECTURE_DECISION.md`
Part 7 Phase 5's "Follow-up: real single-layer engine-level load, done")
was lost to a session reset. The real code/checkpoint/GPU state survived
on two sibling machines reachable over SSH from this session; this probe
was rebuilt and re-executed there rather than guessed at.

## What this proves

The exact real call chain that used to hit both SM75 bugs -
`auto_gptq.py` -> `humming_utils.py` -> `HummingKernel.prepare_kernels`/
`RepackWeightKernel` -> NVRTC compile -> `cuModuleLoad` -> real kernel
launch - now completes cleanly through vLLM's **actual production
model-loading path** (`LLM(...)`, not a hand-rolled repro), on real
Tesla T4 hardware, with `humming_fix.patch` applied.

## Files

- `extract_single_layer.py` — builds a standalone ~4GB single-layer
  checkpoint (real layer-0 GPTQ tensors + embed/norm/lm_head) from the
  real 61GB `gpt-oss-120b-gptq` checkpoint's shard 1. Run on the machine
  holding the real checkpoint; writes to `/gpt-oss-120b-gptq-1layer`
  (on `/`, deliberately **not** `/kaggle/working` - downloading/writing
  large files under `/kaggle/working` can fill its small dedicated
  volume and crash the session, which is what caused the original data
  loss this reconstruction recovered from). Never opens any file under
  the real checkpoint's directory in write mode.
- `single_layer_load_test.py` — real `vllm.LLM(...)` construction against
  that standalone checkpoint, `dtype="float16"` (T4 has no bf16 tensor
  cores), then a real `generate()` call.
- `last_successful_run.log` — full real output of a passing run.

## Real result (this session, 2026-08-03)

```
INFO [auto_gptq.py:249] Layer 'model.layers.0.mlp.experts' is not supported by GPTQMoeMarlin. Falling back to Moe WNA16 kernels.
INFO [int_wna16.py:297] Using 'HUMMING' WNA16 MoE backend.
INFO [default_loader.py:430] Loading weights took 2.61 seconds
INFO [gpu_model_runner.py:5405] Model loading took 3.88 GiB memory and 6.537652 seconds
...
GENERATED TOKEN IDS: [95922, 143590, 79903, 158287]
GENERATED TEXT: ' Combined clasp exchanged Rapport'
```

Gibberish text is the *expected*, correct result — 34 of 36 real
transformer layers are structurally absent by construction in this
single-layer stub. The signal is "real weights loaded, real forward
pass ran, real tokens came out, zero crashes" - not coherence.

## One new environment finding, not in the original writeup

`RepackWeightKernel`'s NVRTC compile failed on first attempt with
`nvrtc: error: failed to open libnvrtc-builtins.so.13.0` even though
that file exists on disk (`nvidia/cu13/lib/` via pip) - it just wasn't
on the dynamic linker's search path. Fixed by prepending that directory
to `LD_LIBRARY_PATH` before running. Environment-specific (CUDA/pip
package layout drift since the original run), unrelated to Bug 1/Bug 2.
