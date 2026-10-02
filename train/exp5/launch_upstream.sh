#!/bin/bash
# MAReasoning Exp 5 (2026-10-02). Usage: launch_upstream.sh <config.yaml> [run-tag]
# Runs the PRISTINE upstream clone (/data/sakhaei/code/AgentFlow-upstream, its own uv venv) the way its README does:
# `python train/rollout.py` (rollout workers) + `python train/train_agent.py` (verl trainer), both reading train/config.yaml.
# Upstream hard-codes that path, so the chosen config is copied over train/config.yaml (the original is train/config.yaml.upstream).
set -u
U=/data/sakhaei/code/AgentFlow-upstream
cd $U; source .venv/bin/activate
CFG=${1:?config yaml}; TAG=${2:-$(date +%Y%m%d-%H%M%S)}
[ "$(readlink -f $CFG)" = "$(readlink -f train/config.yaml)" ] || cp "$CFG" train/config.yaml
LOGD=/data/sakhaei/runs/train/$TAG; mkdir -p "$LOGD"
export HF_HOME=/data/sakhaei/hf PYTHONPATH=$U PYTHONUNBUFFERED=1 VLLM_USE_V1=1 HYDRA_FULL_ERROR=1
unset OPENAI_API_KEY GOOGLE_API_KEY DASHSCOPE_API_KEY GEMINI_API_KEY ANTHROPIC_API_KEY
eval "$(python - <<PY
import yaml, os, shlex
for k, v in yaml.safe_load(open("train/config.yaml"))["env"].items():
    print(f"export {k}={shlex.quote(str(v))}")
PY
)"
echo "config=$CFG tag=$TAG gpus=$CUDA_VISIBLE_DEVICES logs=$LOGD venv=$U/.venv"
ray stop --force >/dev/null 2>&1 || true
export RAY_memory_usage_threshold=0.97
ray start --head --num-gpus $(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l) --object-store-memory 40000000000 --dashboard-host=127.0.0.1 > "$LOGD/ray.log" 2>&1
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MALLOC_ARENA_MAX=2 TOKENIZERS_PARALLELISM=false setsid nohup python train/rollout.py > "$LOGD/rollout.log" 2>&1 < /dev/null &
echo $! > "$LOGD/rollout.pid"
sleep 5
setsid nohup python train/train_agent.py > "$LOGD/train.log" 2>&1 < /dev/null &
echo $! > "$LOGD/train.pid"
echo "started: rollout pid $(cat $LOGD/rollout.pid), trainer pid $(cat $LOGD/train.pid)"
