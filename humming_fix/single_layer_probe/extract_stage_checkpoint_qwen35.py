"""Same idea as extract_stage_checkpoint.py (real per-stage PP checkpoint
extraction) but for Qwen3.5-122B-A10B-GPTQ-Int4's real key layout, which
differs from GPT-OSS's:

  - decoder layers live under "model.language_model.layers.{i}." (not
    "model.layers.{i}.")
  - globals are "lm_head.weight", "model.language_model.embed_tokens.weight",
    "model.language_model.norm.weight"
  - the checkpoint also contains "model.visual.*" (vision tower, ~27 blocks)
    - excluded by default (text-only serving, --language-model-only), kept
    with --include-vision for a stage that will actually serve images/video.
    Only the FIRST PP stage should ever pass --include-vision (it's the one
    that embeds raw input_ids + pixel_values; later stages only see
    forwarded hidden states) - launch that stage WITHOUT
    --language-model-only and every other stage WITH it, or every stage
    will construct+load the full (unquantized) vision tower into VRAM for
    no functional benefit. This is safe/cheap when done right: vLLM's
    language_model_only marks the vision tower as a StageMissingLayer and
    does not require its weights on stages that pass it
    (vllm/model_executor/models/interfaces.py's _mark_tower_model +
    vllm/config/multimodal.py's get_limit_per_prompt returning 0).
  - "mtp.*" (1 extra hidden layer for multi-token prediction / speculative
    decoding) is excluded by default but kept with --include-mtp. Real,
    measured fact: mtp.* tensors live in shards 37-39, the SAME shards the
    globals (embed_tokens/norm/lm_head) already pull in - so on the last
    stage (which needs --include-globals anyway) --include-mtp costs zero
    extra shard downloads. MTP only needs to live on the LAST PP stage: it
    consumes the target model's OWN final hidden_states directly (no
    cross-machine transport of its own), same-process, same-GPU - see
    Qwen3_5MTP's get_pp_group().is_last_rank checks in
    vllm/model_executor/models/qwen3_5_mtp.py.
  - config.json's vision_config block is KEPT as-is (not stripped): the
    Qwen3_5ForConditionalGeneration.__init__ unconditionally reads
    config.vision_config to size self.visual_dim etc. BEFORE the
    language_model_only skip logic applies at weight-load time, so removing
    the config key (not just the weights) would crash __init__.

Writes output in MULTIPLE safetensors part files (real bug hit running
this for real: the original single-`tensors={}`-dict-then-one-save_file()
approach held the whole ~24-26GB stage in RAM at once, and on a 31GB-RAM
machine with a 12-layer+globals+mtp stage, RSS climbed to ~26GB with the
process stuck in D-state - real OOM risk, no swap configured on these
machines). Source shards are processed in small batches (default 4,
~1.5-2GB/shard observed real) and flushed to their own part file
immediately, bounding peak RSS to roughly one batch's worth regardless of
total stage size.

Usage:
    python3 extract_stage_checkpoint_qwen35.py --start 0 --end 16 --out /data/stage0-checkpoint --include-globals
    python3 extract_stage_checkpoint_qwen35.py --start 16 --end 32 --out /data/stage1-checkpoint
    python3 extract_stage_checkpoint_qwen35.py --start 32 --end 48 --out /data/stage2-checkpoint --include-globals
"""
import argparse
import json
import os

from safetensors import safe_open
from safetensors.torch import save_file

SRC_DIR = "/data/models/qwen3.5-122b-a10b-gptq"

LAYER_PREFIX_TMPL = "model.language_model.layers.{i}."
GLOBAL_KEYS = {
    "lm_head.weight",
    "model.language_model.embed_tokens.weight",
    "model.language_model.norm.weight",
}


def main() -> None:
    global SRC_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True, help="first layer index, inclusive")
    ap.add_argument("--end", type=int, required=True, help="last layer index, exclusive")
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default=SRC_DIR)
    ap.add_argument("--include-globals", action="store_true", default=False)
    ap.add_argument("--include-mtp", action="store_true", default=False,
                     help="keep mtp.* tensors for --speculative-config method=mtp "
                          "(only meaningful on the last PP stage)")
    ap.add_argument("--include-vision", action="store_true", default=False,
                     help="keep model.visual.* (vision tower) tensors - only meaningful "
                          "on the FIRST PP stage (the one that embeds raw input_ids + "
                          "pixel_values; later stages only ever see forwarded hidden "
                          "states, never raw pixels). Pair with launching that stage "
                          "WITHOUT --language-model-only, and every other stage WITH it "
                          "(language_model_only=True makes get_limit_per_prompt() return "
                          "0 for every modality, which _mark_tower_model uses to skip "
                          "constructing/loading self.visual entirely on that process - "
                          "see vllm/config/multimodal.py and "
                          "vllm/model_executor/models/interfaces.py). Without this "
                          "asymmetry, every stage would load the full (unquantized) "
                          "vision tower into VRAM for no functional benefit.")
    ap.add_argument("--shards-per-batch", type=int, default=4,
                     help="source shards to hold in RAM at once before flushing to a "
                          "part file - lower this on low-RAM machines")
    args = ap.parse_args()
    SRC_DIR = args.src

    assert not args.out.startswith("/kaggle/working"), "large checkpoint output must not go under /kaggle/working"
    os.makedirs(args.out, exist_ok=True)

    with open(os.path.join(SRC_DIR, "model.safetensors.index.json")) as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]

    layer_prefixes = tuple(LAYER_PREFIX_TMPL.format(i=i) for i in range(args.start, args.end))

    keep_keys = [k for k in weight_map if k.startswith(layer_prefixes)]
    if args.include_globals:
        keep_keys += [k for k in weight_map if k in GLOBAL_KEYS]
    if args.include_mtp:
        keep_keys += [k for k in weight_map if k.startswith("mtp.")]
    if args.include_vision:
        keep_keys += [k for k in weight_map if k.startswith("model.visual.")]

    n_visual = sum(1 for k in weight_map if k.startswith("model.visual."))
    n_mtp = sum(1 for k in weight_map if k.startswith("mtp."))
    print(f"{'including' if args.include_vision else 'excluding'} {n_visual} vision-tower tensors "
          f"{'and including' if args.include_mtp else 'and excluding'} {n_mtp} mtp tensors")
    print(f"keeping {len(keep_keys)} tensors for layers [{args.start}, {args.end})"
          f"{' + globals' if args.include_globals else ''}"
          f"{' + mtp' if args.include_mtp else ''}"
          f"{' + vision' if args.include_vision else ''}")

    needed_shards = sorted(set(weight_map[k] for k in keep_keys))
    print(f"reading from {len(needed_shards)} shard(s) in batches of {args.shards_per_batch}")

    new_weight_map: dict[str, str] = {}
    total_bytes = 0
    part_num = 0
    for batch_start in range(0, len(needed_shards), args.shards_per_batch):
        batch_shards = needed_shards[batch_start:batch_start + args.shards_per_batch]
        batch_tensors = {}
        for shard in batch_shards:
            shard_path = os.path.join(SRC_DIR, shard)
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
        print(f"wrote {part_name}: {len(batch_tensors)} tensors, {part_bytes / 1e9:.2f} GB "
              f"(shards {batch_start + 1}-{batch_start + len(batch_shards)}/{len(needed_shards)})")
        for k in batch_tensors:
            new_weight_map[k] = part_name
        del batch_tensors

    print(f"total stage checkpoint size: {total_bytes / 1e9:.2f} GB across {part_num} part file(s)")

    new_index = {"metadata": {"total_size": total_bytes}, "weight_map": new_weight_map}
    with open(os.path.join(args.out, "model.safetensors.index.json"), "w") as f:
        json.dump(new_index, f, indent=2)

    with open(os.path.join(SRC_DIR, "config.json")) as f:
        config = json.load(f)
    # num_hidden_layers / layer_types live under text_config for the real
    # (multimodal) checkpoint. Keep both the top-level and text_config views
    # of layer count UNCHANGED (real PP-stage mode - see
    # extract_stage_checkpoint.py's --renumber docstring for why: vLLM's
    # make_layers() slices [start_layer,end_layer) using pp_rank/world_size
    # against the TRUE global num_hidden_layers).
    if args.include_mtp:
        # Real bug hit TWICE now on two different checkpoints of this same
        # architecture family (first on an earlier Akun2 deployment, again
        # here): the source config.json's quantization_config.dynamic marks
        # ALL "mtp.*" tensors as GPTQ-quantized ("+:.*mtp.*"), but
        # mtp.*.mlp.down_proj is NOT actually quantized in the checkpoint -
        # it has a plain .weight tensor, not
        # .qweight/.qzeros/.scales/.g_idx. Without this exclusion,
        # AutoGPTQLinearMethod's weight loader looks for
        # 'layers.0.mlp.down_proj.weight' and errors with "no module or
        # parameter named ... available parameters ... {qweight, qzeros,
        # scales, g_idx}" the moment the drafter model tries to load it.
        # Adding a more-specific "-:" (exclude) rule AFTER the general
        # "+:.*mtp.*" one fixes it (confirmed working this same way on an
        # earlier Akun2 deployment) - appended, not prepended, to actually
        # override the general rule for this one case.
        dynamic = config.get("quantization_config", {}).get("dynamic")
        if dynamic is not None:
            dynamic[r"-:.*mtp.*mlp\.down_proj.*"] = {}
    print(f"wrote config.json (REAL PP-STAGE mode: "
          f"num_hidden_layers={config['text_config']['num_hidden_layers']} unchanged, "
          f"original layer numbering [{args.start},{args.end}) kept, vision_config kept for __init__)")
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    for fname in [
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "generation_config.json", "chat_template.jinja", "chat_template.json",
        "preprocessor_config.json", "video_preprocessor_config.json", "merges.txt", "vocab.json",
    ]:
        src = os.path.join(SRC_DIR, fname)
        if os.path.exists(src):
            with open(src, "rb") as fsrc, open(os.path.join(args.out, fname), "wb") as fdst:
                fdst.write(fsrc.read())

    print("\nDONE:", args.out)


if __name__ == "__main__":
    main()
