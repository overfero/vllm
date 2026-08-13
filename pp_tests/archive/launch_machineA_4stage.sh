#!/bin/bash
cd /kaggle/working/vllm
python3 -u scripts/stage_server.py \
  --model /data/stage0-checkpoint \
  --tensor-parallel-size 2 \
  --pp-rank 0 --pp-world-size 4 \
  --self-name MachineA --next-name MachineB --driver-name MachineD \
  --transport udp --signaling-url https://gt5xhei28qbx.share.zrok.io \
  --transport-connect-timeout 900 \
  --quantization gptq --dtype float16 --language-model-only \
  --max-model-len 8192 --gpu-memory-utilization 0.95 \
  --num-gpu-blocks-override 60 --max-num-seqs 8 --enable-cudagraph
