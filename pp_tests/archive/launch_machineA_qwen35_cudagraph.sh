#!/bin/bash
cd /kaggle/working/vllm
python3 -u scripts/stage_server.py \
  --model /data/stage0-checkpoint \
  --tensor-parallel-size 2 \
  --pp-rank 0 --pp-world-size 3 \
  --self-name MachineA --next-name MachineB --driver-name MachineC \
  --transport udp --signaling-url https://zaae6g84q41o.share.zrok.io \
  --quantization gptq --dtype float16 --language-model-only \
  --max-model-len 8192 --gpu-memory-utilization 0.95 \
  --num-gpu-blocks-override 60 --enable-cudagraph
