"""Combined selective-download + extract for a Qwen3.5-122B-A10B-GPTQ-Int4
PP-stage checkpoint. Unlike GPT-OSS (extract_stage_checkpoint.py), this does
NOT download the full 78.8GB checkpoint first: Qwen3.5's checkpoint is
scattered across 39 shards with no clean per-layer partition, but each
16-layer PP stage still only touches 24-26 of the 39 shards (~53-57GB
instead of 78.8GB) - real, measured on the machine this was built on. This
matters because Qwen3.5's checkpoint is bigger than GPT-OSS's (78.8GB vs
61GB), so downloading it in full first would leave much less disk headroom
during the brief moment both the raw download and the extracted stage
shard exist on disk (see memory/feedback_kaggle_working_placement.md).

Steps:
  1. Download just the small metadata files (config.json,
     model.safetensors.index.json, tokenizer/*) - needed to compute which
     shards this stage's layer range actually touches.
  2. Parse the index to find the minimal shard set for [start, end) (+
     globals if requested).
  3. `hf download --include <those shards + metadata>` - selective, not
     the full checkpoint.
  4. Run the same tensor-level extraction extract_stage_checkpoint_qwen35.py
     does (import its function directly, no subprocess).
  5. Delete the downloaded (partial) checkpoint dir, unless
     --keep-full-checkpoint.

Usage:
    python3 download_and_extract_qwen35_stage.py --start 0 --end 16 \
        --out /data/stage0-checkpoint --include-globals \
        --checkpoint-dir /data/models/qwen3.5-122b-a10b-gptq
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

DEFAULT_REPO = "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4"
DEFAULT_NUM_HIDDEN_LAYERS = 48
LAYER_PREFIX_TMPL = "model.language_model.layers.{i}."
GLOBAL_KEYS = {
    "lm_head.weight",
    "model.language_model.embed_tokens.weight",
    "model.language_model.norm.weight",
}
META_FILES = [
    "config.json", "model.safetensors.index.json", "tokenizer.json",
    "tokenizer_config.json", "generation_config.json", "chat_template.jinja",
    "merges.txt", "vocab.json", "preprocessor_config.json",
    "video_preprocessor_config.json",
]


def hf_download(repo: str, checkpoint_dir: str, includes: list[str]) -> None:
    cmd = ["hf", "download", repo, "--local-dir", checkpoint_dir]
    for pat in includes:
        cmd += ["--include", pat]
    print(f"downloading {len(includes)} file patterns to {checkpoint_dir}...")
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=DEFAULT_REPO,
                     help="any GPTQ checkpoint sharing Qwen3.5's key layout - real, "
                          "measured working example: Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ "
                          "(architectures=Qwen3_5ForConditionalGeneration, 64 layers)")
    ap.add_argument("--num-layers", type=int, default=DEFAULT_NUM_HIDDEN_LAYERS)
    ap.add_argument("--checkpoint-dir", default="/data/models/qwen3.5-122b-a10b-gptq")
    ap.add_argument("--include-globals", action="store_true", default=False)
    ap.add_argument("--include-mtp", action="store_true", default=False,
                     help="also pull mtp.* for --speculative-config method=mtp - real, "
                          "measured fact: mtp.* lives in shards 37-39, the same shards "
                          "--include-globals already needs, so this is usually free")
    ap.add_argument("--keep-full-checkpoint", action="store_true", default=False)
    ap.add_argument("--shards-per-batch", type=int, default=4,
                     help="passed through to extract_stage_checkpoint_qwen35.py - lower "
                          "this on low-RAM machines (real OOM risk hit at the default on "
                          "a 31GB-RAM machine extracting a 12-layer+globals+mtp stage)")
    args = ap.parse_args()

    assert not args.out.startswith("/kaggle/working"), "large checkpoint output must not go under /kaggle/working"
    assert 0 <= args.start < args.end <= args.num_layers

    # step 1: metadata only
    hf_download(args.repo, args.checkpoint_dir, META_FILES)

    # step 2: compute needed shards
    with open(os.path.join(args.checkpoint_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]

    layer_prefixes = tuple(LAYER_PREFIX_TMPL.format(i=i) for i in range(args.start, args.end))
    keep_keys = [k for k in weight_map if k.startswith(layer_prefixes)]
    if args.include_globals:
        keep_keys += [k for k in weight_map if k in GLOBAL_KEYS]
    if args.include_mtp:
        keep_keys += [k for k in weight_map if k.startswith("mtp.")]
    needed_shards = sorted(set(weight_map[k] for k in keep_keys))
    total_shards = len(set(weight_map.values()))
    print(f"stage [{args.start},{args.end}) needs {len(needed_shards)}/{total_shards} shards")

    # step 3: selective download of just those shards
    hf_download(args.repo, args.checkpoint_dir, needed_shards)

    # step 4: extract (reuse the same logic as extract_stage_checkpoint_qwen35.py)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import extract_stage_checkpoint_qwen35 as extractor

    sys.argv = [
        "extract_stage_checkpoint_qwen35.py",
        "--start", str(args.start), "--end", str(args.end),
        "--out", args.out, "--src", args.checkpoint_dir,
    ]
    if args.include_globals:
        sys.argv.append("--include-globals")
    if args.include_mtp:
        sys.argv.append("--include-mtp")
    sys.argv += ["--shards-per-batch", str(args.shards_per_batch)]
    extractor.main()

    # step 5: cleanup
    if not args.keep_full_checkpoint:
        print(f"deleting {args.checkpoint_dir} to keep disk footprint small")
        shutil.rmtree(args.checkpoint_dir, ignore_errors=True)
    else:
        print(f"--keep-full-checkpoint set, leaving {args.checkpoint_dir} in place")

    print("\nDONE:", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
