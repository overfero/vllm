#!/bin/bash
: "${SIGNALING_URL:?set SIGNALING_URL to the orchestrator signaling server URL, e.g. https://your-tunnel.example.com}"
cd /kaggle/working/vllm
# Custom asymmetric PP split (16/16/12/4 layers across A/B/C/D) - without
# this, vLLM's get_pp_indices() silently falls back to an EVEN division
# (48/4=12 per stage) regardless of what layers each stage's checkpoint
# actually contains, causing each stage to load the wrong layer range
# (missing layers left randomly-initialized -> NaN propagation). Every
# stage must be given the SAME full partition list; each stage derives
# its own [start,end) from its own --pp-rank against this list.
export VLLM_PP_LAYER_PARTITION="16,16,12,4"
# Every stage must agree on the same KV-cache group_size (derived from the
# FULL 48-layer model's full_attention:linear_attention ratio = 12:36, so
# group_size=12) - see kv_cache_utils.py's VLLM_KV_CACHE_GROUP_SIZE_OVERRIDE
# comment for why an asymmetric split otherwise makes each independent
# per-machine engine derive a DIFFERENT group_size from its own local layer
# count, breaking the driver's centrally-scheduled block_ids shape.
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
  --num-gpu-blocks-override 30 --max-num-seqs 8 \
  --cpu-offload-gb 3 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'
