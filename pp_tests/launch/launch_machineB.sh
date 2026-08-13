#!/bin/bash
: "${SIGNALING_URL:?set SIGNALING_URL to the orchestrator signaling server URL, e.g. https://your-tunnel.example.com}"
cd /kaggle/working/vllm
python3 -u scripts/stage_server.py \
  --model /data/stage1-checkpoint \
  --tensor-parallel-size 2 \
  --pp-rank 1 --pp-world-size 4 \
  --self-name MachineB --prev-name MachineA --next-name MachineC --driver-name MachineD \
  --transport udp --signaling-url "$SIGNALING_URL" \
  --transport-connect-timeout 300 \
  --quantization gptq --dtype float16 --language-model-only \
  --max-model-len 8192 --gpu-memory-utilization 0.95 \
  --num-gpu-blocks-override 60 --max-num-seqs 8 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'
