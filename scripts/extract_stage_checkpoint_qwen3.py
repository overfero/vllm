"""Real per-stage PP checkpoint extraction for plain Qwen3 (dense, e.g.
Qwen3-1.7B-Base) - same idea and same memory-bounded batched-write
approach as `humming_fix/single_layer_probe/extract_stage_checkpoint_qwen35.py`
(that one's for Qwen3.5-122B-A10B-GPTQ-Int4's real key layout, which is
NOT the same as plain Qwen3's - verified directly, not assumed, by
reading Qwen3-1.7B-Base's own `model.safetensors` header and
`config.json`, not copied from the Qwen3.5 script's docstring):

  - decoder layers live under "model.layers.{i}." (NOT
    "model.language_model.layers.{i}." - that nested prefix is
    Qwen3.5's multimodal wrapper structure, plain Qwen3 has no such
    nesting).
  - globals are "model.embed_tokens.weight" and "model.norm.weight" -
    there is NO separate "lm_head.weight" key at all when
    `config.tie_word_embeddings` is true (confirmed: Qwen3-1.7B-Base's
    real config.json has `"tie_word_embeddings": true`, and its real
    safetensors header has no "lm_head.weight" entry).
  - Real, verified-by-reading-vllm's-own-source finding (`vllm/
    model_executor/models/qwen2.py:363-366`, which `Qwen3Model`
    subclasses unchanged): when `tie_word_embeddings` is true, BOTH the
    FIRST PP stage (for input embedding) AND the LAST PP stage (to serve
    as `lm_head` - `Qwen3ForCausalLM` sets `self.lm_head =
    self.model.embed_tokens` on the last rank, see qwen3.py:297-299) each
    independently construct their OWN `embed_tokens` module and need
    "model.embed_tokens.weight" in THEIR OWN checkpoint shard - there is
    no cross-process sharing of that tensor the way a single-process PP
    setup could get away with. `--include-embed` is a plain "keep this
    tensor" flag (no lm_head-specific behavior) - pass it on whichever
    stage(s) actually need the tensor: always the first stage, and also
    the last stage when tie_word_embeddings is true.
  - No quantization/MTP/vision special-casing here - plain Qwen3 has
    none of those. If a future target model needs any of them, port the
    relevant handling from the Qwen3.5 script rather than guessing it
    applies unchanged here.

Supports either a single `model.safetensors` file OR a sharded
`model-*.safetensors` + `model.safetensors.index.json` source (Qwen3-
1.7B-Base is the single-file case; kept general so a larger future model
in this same architecture family - the project's own stated next step
after the small model works - doesn't need a second script).

Usage (2-stage example, tie_word_embeddings=true):
    python3 extract_stage_checkpoint_qwen3.py --start 0 --end 14 --out /data/qwen3-1.7b-stage0 --src /data/models/qwen3-1.7b-base --include-embed
    python3 extract_stage_checkpoint_qwen3.py --start 14 --end 28 --out /data/qwen3-1.7b-stage1 --src /data/models/qwen3-1.7b-base --include-embed --include-norm
"""
import argparse
import json
import os

from safetensors import safe_open
from safetensors.torch import save_file

LAYER_PREFIX_TMPL = "model.layers.{i}."
EMBED_KEY = "model.embed_tokens.weight"
NORM_KEY = "model.norm.weight"


def _load_weight_map(src_dir: str) -> tuple[dict[str, str], bool]:
    """Returns (weight_map: tensor_name -> shard_filename, is_single_file).
    Builds a synthetic one-shard weight_map when the source is a single
    `model.safetensors` file (no index.json exists at all in that case -
    confirmed directly for Qwen3-1.7B-Base, not assumed)."""
    index_path = os.path.join(src_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            return json.load(f)["weight_map"], False

    single_path = os.path.join(src_dir, "model.safetensors")
    if not os.path.exists(single_path):
        raise FileNotFoundError(
            f"neither model.safetensors.index.json nor model.safetensors found under {src_dir}"
        )
    with safe_open(single_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
    return {k: "model.safetensors" for k in keys}, True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True, help="first layer index, inclusive")
    ap.add_argument("--end", type=int, required=True, help="last layer index, exclusive")
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", required=True, help="local directory containing the full checkpoint")
    ap.add_argument("--include-embed", action="store_true", default=False,
                     help="keep model.embed_tokens.weight - the FIRST PP stage always needs this (input "
                          "embedding); the LAST stage ALSO needs it when tie_word_embeddings=true (it "
                          "serves as lm_head there - see this file's own module docstring for the real "
                          "vllm source citation, qwen2.py:363-366/qwen3.py:297-299). Pass on whichever "
                          "stage(s) actually need it - do not pass it on middle stages, which need neither.")
    ap.add_argument("--include-norm", action="store_true", default=False,
                     help="keep model.norm.weight - only the LAST stage needs this (final layernorm "
                          "immediately before the lm_head projection).")
    ap.add_argument("--shards-per-batch", type=int, default=4,
                     help="source shards to hold in RAM at once before flushing to a part file - lower "
                          "this on low-RAM machines. Irrelevant (always 1) for a single-file source.")
    args = ap.parse_args()

    assert not args.out.startswith("/kaggle/working"), "large checkpoint output must not go under /kaggle/working"
    os.makedirs(args.out, exist_ok=True)

    weight_map, is_single_file = _load_weight_map(args.src)

    layer_prefixes = tuple(LAYER_PREFIX_TMPL.format(i=i) for i in range(args.start, args.end))
    keep_keys = [k for k in weight_map if k.startswith(layer_prefixes)]
    if args.include_embed and EMBED_KEY in weight_map:
        keep_keys.append(EMBED_KEY)
    if args.include_norm and NORM_KEY in weight_map:
        keep_keys.append(NORM_KEY)

    print(f"source is {'a single safetensors file' if is_single_file else 'sharded'}")
    print(f"keeping {len(keep_keys)} tensors for layers [{args.start}, {args.end})"
          f"{' + embed_tokens' if args.include_embed and EMBED_KEY in keep_keys else ''}"
          f"{' + norm' if args.include_norm and NORM_KEY in keep_keys else ''}")

    needed_shards = sorted(set(weight_map[k] for k in keep_keys))
    print(f"reading from {len(needed_shards)} shard(s)/file(s) in batches of {args.shards_per_batch}")

    new_weight_map: dict[str, str] = {}
    total_bytes = 0
    part_num = 0
    for batch_start in range(0, len(needed_shards), args.shards_per_batch):
        batch_shards = needed_shards[batch_start:batch_start + args.shards_per_batch]
        batch_tensors = {}
        for shard in batch_shards:
            shard_path = os.path.join(args.src, shard)
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for k in keep_keys:
                    if weight_map[k] == shard:
                        batch_tensors[k] = f.get_tensor(k)
        if not batch_tensors:
            continue
        part_num += 1
        part_name = f"model-{part_num:05d}.safetensors"
        part_bytes = sum(t.numel() * t.element_size() for t in batch_tensors.values())
        total_bytes += part_bytes
        save_file(batch_tensors, os.path.join(args.out, part_name), metadata={"format": "pt"})
        print(f"wrote {part_name}: {len(batch_tensors)} tensors, {part_bytes / 1e9:.3f} GB")
        for k in batch_tensors:
            new_weight_map[k] = part_name
        del batch_tensors

    print(f"total stage checkpoint size: {total_bytes / 1e9:.3f} GB across {part_num} part file(s)")

    new_index = {"metadata": {"total_size": total_bytes}, "weight_map": new_weight_map}
    with open(os.path.join(args.out, "model.safetensors.index.json"), "w") as f:
        json.dump(new_index, f, indent=2)

    with open(os.path.join(args.src, "config.json")) as f:
        config = json.load(f)
    # num_hidden_layers kept UNCHANGED (real PP-stage mode) - vLLM's
    # make_layers() slices [start_layer, end_layer) using pp_rank/
    # world_size against the TRUE global num_hidden_layers, same
    # reasoning as the Qwen3.5 extractor's identical choice.
    tie = config.get("tie_word_embeddings", False)
    if not tie and "lm_head.weight" in weight_map and args.include_norm and not args.include_embed:
        print(
            "WARNING: tie_word_embeddings=false for this checkpoint, meaning there's a real, separate "
            "lm_head.weight tensor this script does not currently handle (no target model has needed it "
            "yet - the real, verified case built here is the tied-embedding one, see this file's own "
            "module docstring). The stage being written now looks like a last-stage call (--include-norm) "
            "but did not pass --include-embed - if this checkpoint isn't tied, it also needs an explicit "
            "lm_head.weight, not provided by any flag here yet. Do not assume this stage is complete."
        )
    print(f"wrote config.json (num_hidden_layers={config['num_hidden_layers']} unchanged, "
          f"original layer numbering [{args.start},{args.end}) kept, tie_word_embeddings={tie})")
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    for fname in [
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "generation_config.json", "merges.txt", "vocab.json",
    ]:
        src = os.path.join(args.src, fname)
        if os.path.exists(src):
            with open(src, "rb") as fsrc, open(os.path.join(args.out, fname), "wb") as fdst:
                fdst.write(fsrc.read())

    print("\nDONE:", args.out)


if __name__ == "__main__":
    main()
