#!/bin/bash
set -e
mkdir -p /data/models
echo "[stage0] downloading selective shards..."
hf download Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 --local-dir /data/models/qwen3.5-122b-a10b-gptq --include "model.safetensors-00001-of-00039.safetensors" --include "model.safetensors-00002-of-00039.safetensors" --include "model.safetensors-00003-of-00039.safetensors" --include "model.safetensors-00004-of-00039.safetensors" --include "model.safetensors-00009-of-00039.safetensors" --include "model.safetensors-00012-of-00039.safetensors" --include "model.safetensors-00014-of-00039.safetensors" --include "model.safetensors-00017-of-00039.safetensors" --include "model.safetensors-00019-of-00039.safetensors" --include "model.safetensors-00021-of-00039.safetensors" --include "model.safetensors-00022-of-00039.safetensors" --include "model.safetensors-00023-of-00039.safetensors" --include "model.safetensors-00025-of-00039.safetensors" --include "model.safetensors-00026-of-00039.safetensors" --include "model.safetensors-00029-of-00039.safetensors" --include "model.safetensors-00030-of-00039.safetensors" --include "model.safetensors-00031-of-00039.safetensors" --include "model.safetensors-00033-of-00039.safetensors" --include "model.safetensors-00034-of-00039.safetensors" --include "model.safetensors-00035-of-00039.safetensors" --include "model.safetensors-00036-of-00039.safetensors" --include "model.safetensors-00037-of-00039.safetensors" --include "model.safetensors-00038-of-00039.safetensors" --include "model.safetensors-00039-of-00039.safetensors" --include "config.json" --include "model.safetensors.index.json" --include "tokenizer.json" --include "tokenizer_config.json" --include "generation_config.json" --include "chat_template.jinja" --include "merges.txt" --include "vocab.json" --include "preprocessor_config.json" --include "video_preprocessor_config.json" 2>&1 | tail -30
echo "[stage0] download done, size:"
du -sh /data/models/qwen3.5-122b-a10b-gptq
echo "[stage0] extracting stage checkpoint..."
cd /kaggle/working/humming_fix/single_layer_probe
python3 extract_stage_checkpoint_qwen35.py --start 0 --end 16 --out /data/stage0-checkpoint --src /data/models/qwen3.5-122b-a10b-gptq --include-globals
echo "[stage0] deleting full download to free disk..."
rm -rf /data/models/qwen3.5-122b-a10b-gptq
echo "[stage0] DONE"
du -sh /data/stage0-checkpoint
