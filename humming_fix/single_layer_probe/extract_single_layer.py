"""Build a standalone single-layer GPT-OSS-120B GPTQ checkpoint from the
real 61GB checkpoint's layer 0 + global tensors, to real-load through
vLLM's actual production model-loading path with the Humming MoE backend.

Run on the machine that has the real checkpoint (not the code-restore
machine) - reads from /data/models/gpt-oss-120b-gptq (real, untouched,
read-only here), writes the new standalone checkpoint to
/gpt-oss-120b-gptq-1layer (on `/`, NOT /kaggle/working - large-file rule).

Safety note (a real mistake happened here in the original session and is
deliberately avoided this time): every write target below is a fresh
path under /gpt-oss-120b-gptq-1layer, never a symlink and never a path
under the real checkpoint directory. Tensor data is read via
safetensors.safe_open (read-only) and written fresh with
safetensors.torch.save_file - no file under /data/models is ever opened
in write mode.
"""
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC_DIR = "/data/models/gpt-oss-120b-gptq"
DST_DIR = "/gpt-oss-120b-gptq-1layer"
SRC_SHARD = os.path.join(SRC_DIR, "model-00001-of-00016.safetensors")

os.makedirs(DST_DIR, exist_ok=True)

with open(os.path.join(SRC_DIR, "model.safetensors.index.json")) as f:
    idx = json.load(f)
weight_map = idx["weight_map"]

keep_keys = [
    k for k in weight_map
    if k.startswith("model.layers.0.") or k in ("lm_head.weight", "model.embed_tokens.weight", "model.norm.weight")
]
print(f"keeping {len(keep_keys)} tensors, all from {SRC_SHARD}")
assert all(weight_map[k] == "model-00001-of-00016.safetensors" for k in keep_keys)

tensors = {}
with safe_open(SRC_SHARD, framework="pt", device="cpu") as f:
    for k in keep_keys:
        tensors[k] = f.get_tensor(k)

total_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
print(f"total single-layer checkpoint size: {total_bytes / 1e9:.2f} GB")

out_safetensors = os.path.join(DST_DIR, "model.safetensors")
save_file(tensors, out_safetensors, metadata={"format": "pt"})
print(f"wrote {out_safetensors}")

new_weight_map = {k: "model.safetensors" for k in keep_keys}
new_index = {
    "metadata": {"total_size": total_bytes},
    "weight_map": new_weight_map,
}
with open(os.path.join(DST_DIR, "model.safetensors.index.json"), "w") as f:
    json.dump(new_index, f, indent=2)
print("wrote index")

with open(os.path.join(SRC_DIR, "config.json")) as f:
    config = json.load(f)
config["num_hidden_layers"] = 1
config["layer_types"] = [config["layer_types"][0]]
with open(os.path.join(DST_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)
print("wrote config.json (num_hidden_layers=1)")

for fname in [
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "generation_config.json", "chat_template.jinja", "chat_template.json",
]:
    src = os.path.join(SRC_DIR, fname)
    if os.path.exists(src):
        with open(src, "rb") as fsrc, open(os.path.join(DST_DIR, fname), "wb") as fdst:
            fdst.write(fsrc.read())
        print(f"copied {fname}")

print("\nDONE:", DST_DIR)
print(os.listdir(DST_DIR))
