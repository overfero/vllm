#!/bin/bash
: "${SIGNALING_URL:?set SIGNALING_URL to the orchestrator signaling server URL, e.g. https://your-tunnel.example.com}"
cd /kaggle/working/vllm
export VLLM_PP_LAYER_PARTITION="12,12,12,12"
export VLLM_KV_CACHE_GROUP_SIZE_OVERRIDE="12"
python3 scripts/launch_pp_stage.py \
  --pp-rank 3 --pp-world-size 4 \
  --self-name MachineD --prev-name MachineC \
  --transport udp --signaling-url "$SIGNALING_URL" \
  --transport-connect-timeout 900 \
  --model /data/stage3-checkpoint \
  --tensor-parallel-size 2 --dtype float16 --quantization gptq \
  --gpu-memory-utilization 0.95 --language-model-only --max-model-len 8192 \
  --num-gpu-blocks-override 60 --max-num-seqs 8 \
  --enable-cudagraph --cpu-offload-gb 1 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
  --serve --host 0.0.0.0 --port 8080 \
  --remote-stage-names MachineA,MachineB,MachineC
