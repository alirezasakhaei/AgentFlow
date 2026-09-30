#!/bin/bash
# PATCHED (MAReasoning) 2026-09-23. Usage: train/launch_mds.sh <config.yaml> [run-tag]
# Starts the rollout workers (train/rollout.py) and the verl Flow-GRPO trainer (train/train_agent.py)
# from .venv-train, detached, logs under /data/sakhaei/runs/train/<tag>/. Stop with train/stop_mds.sh.
set -u
cd /data/sakhaei/code/AgentFlow
source .venv-train/bin/activate
export AGENTFLOW_TRAIN_CONFIG=${1:?config yaml required}
TAG=${2:-$(date +%Y%m%d-%H%M%S)}
LOGD=/data/sakhaei/runs/train/$TAG; mkdir -p "$LOGD"
export HF_HOME=/data/sakhaei/hf PYTHONPATH=/data/sakhaei/code/AgentFlow PYTHONUNBUFFERED=1
export VLLM_USE_V1=1 HYDRA_FULL_ERROR=1
# Run judge4 died at step 21 with OOM in the optimizer step after 20 clean steps (allocator fragmentation
# creeping up under param/activation offload); expandable segments is verl's standard remedy.
# (PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is NOT usable here: vLLM's sleep-mode cumem allocator asserts against it, 2026-09-24)
unset OPENAI_API_KEY GOOGLE_API_KEY DASHSCOPE_API_KEY GEMINI_API_KEY ANTHROPIC_API_KEY
# Export the config env block BEFORE ray start: Ray workers inherit the raylet's environment, not the
# driver's, so WANDB_MODE/HF_HOME/etc. must already be set when the head node comes up.
eval "$(python - <<EOF
import yaml, os, shlex
for k, v in yaml.safe_load(open(os.environ["AGENTFLOW_TRAIN_CONFIG"]))["env"].items():
    print(f"export {k}={shlex.quote(str(v))}")
EOF
)"
CUDA_DEVICES="$CUDA_VISIBLE_DEVICES"
echo "config=$AGENTFLOW_TRAIN_CONFIG tag=$TAG gpus=$CUDA_VISIBLE_DEVICES logs=$LOGD"
ray stop --force >/dev/null 2>&1 || true
# Host RAM (2026-09-25): with the actor on FSDP2 CPU offload the node sat at Ray's 95% kill threshold of
# 540 GB and four restarts died of "worker(s) killed due to the node running low on memory". Ray's
# object store defaults to 30% of RAM (~160 GB in /dev/shm); cap it, and raise the kill threshold a little.
export RAY_memory_usage_threshold=0.97
ray start --head --num-gpus $(echo "$CUDA_DEVICES" | tr ',' '\n' | wc -l) --object-store-memory 40000000000 --dashboard-host=127.0.0.1 > "$LOGD/ray.log" 2>&1
# Rollout workers: each loads a small CPU embedder (Wikipedia tool) and torch would spawn 96 threads per
# process; 48 workers pushed the load average past 1000 and starved everything. Two threads each.
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MALLOC_ARENA_MAX=2 TOKENIZERS_PARALLELISM=false setsid nohup python train/rollout.py > "$LOGD/rollout.log" 2>&1 < /dev/null &
echo $! > "$LOGD/rollout.pid"
sleep 5
setsid nohup python train/train_agent.py > "$LOGD/train.log" 2>&1 < /dev/null &
echo $! > "$LOGD/train.pid"
echo "started: rollout pid $(cat $LOGD/rollout.pid), trainer pid $(cat $LOGD/train.pid)"
