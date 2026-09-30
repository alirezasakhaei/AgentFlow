#!/bin/bash
cd /data/sakhaei/code/AgentFlow
source .venv/bin/activate
export HF_HOME=/data/sakhaei/hf
L=/data/sakhaei/runs/agentflow-eval
CUDA_VISIBLE_DEVICES=0 nohup vllm serve AgentFlow/agentflow-planner-7b \
  --host 0.0.0.0 --port 8000 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 --max-model-len 16384 \
  > $L/vllm_planner.log 2>&1 &
echo "planner pid $!"
CUDA_VISIBLE_DEVICES=1 nohup vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8001 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 --max-model-len 16384 \
  > $L/vllm_executor.log 2>&1 &
echo "executor pid $!"
