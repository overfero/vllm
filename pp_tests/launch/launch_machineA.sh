#!/bin/bash
: "${SIGNALING_URL:?set SIGNALING_URL to the orchestrator signaling server URL, e.g. https://your-tunnel.example.com}"
cd /kaggle/working/vllm
# Uniform 12/12/12/12 split (back from the 16/16/12/4 asymmetric-split
# experiment - see docs/DEPLOYMENT.md's "Asymmetric PP splits" section for
# why an uneven split needs extra care this uniform one doesn't). Kept
# explicit rather than relying on vLLM's default even-division fallback,
# since that fallback is exactly what silently breaks for an uneven split -
# being explicit here avoids ever being surprised by it again.
export VLLM_PP_LAYER_PARTITION="12,12,12,12"
# Harmless to keep even for a uniform split - every stage naturally derives
# the same group_size (3) from its own local layer counts anyway when the
# split is uniform, so this override (12, from the full 48-layer model) just
# produces a different but still globally-consistent group count. See
# kv_cache_utils.py's VLLM_KV_CACHE_GROUP_SIZE_OVERRIDE comment.
export VLLM_KV_CACHE_GROUP_SIZE_OVERRIDE="12"
python3 -u scripts/stage_server.py \
  --model /data/stage0-checkpoint \
  --tensor-parallel-size 2 \
  --pp-rank 0 --pp-world-size 4 \
  --self-name MachineA --next-name MachineB --driver-name MachineD \
  --transport udp --signaling-url "$SIGNALING_URL" \
  --transport-connect-timeout 900 \
  --quantization gptq --dtype float16 --language-model-only \
  --max-model-len 8192 --gpu-memory-utilization 0.95 \
  --num-gpu-blocks-override 60 --max-num-seqs 8 \
  --enable-cudagraph --cpu-offload-gb 1 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'
