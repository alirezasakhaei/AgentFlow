#!/bin/bash
# PATCHED (MAReasoning). Restart ONLY the rollout workers of a running training run (trainer untouched).
# Usage: train/restart_workers.sh <config.yaml> <tag>
set -u
cd /data/sakhaei/code/AgentFlow; source .venv-train/bin/activate
export AGENTFLOW_TRAIN_CONFIG=${1:?config}; TAG=${2:?tag}; LOGD=/data/sakhaei/runs/train/$TAG
eval "$(python - <<EOF
import yaml, os, shlex
for k, v in yaml.safe_load(open(os.environ["AGENTFLOW_TRAIN_CONFIG"]))["env"].items():
    print(f"export {k}={shlex.quote(str(v))}")
EOF
)"
export HF_HOME=/data/sakhaei/hf PYTHONPATH=/data/sakhaei/code/AgentFlow PYTHONUNBUFFERED=1
unset OPENAI_API_KEY GOOGLE_API_KEY DASHSCOPE_API_KEY GEMINI_API_KEY ANTHROPIC_API_KEY
if [ -f "$LOGD/rollout.pid" ]; then sid=$(ps -o sid= -p "$(cat $LOGD/rollout.pid)" 2>/dev/null | tr -d " "); [ -n "$sid" ] && pkill -TERM -s "$sid"; fi
pkill -TERM AgentFlow 2>/dev/null; sleep 4; pkill -KILL AgentFlow 2>/dev/null
echo "workers left: $(pgrep -c AgentFlow)"
AGENTFLOW_SKIP_PORT_CLEANUP=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MALLOC_ARENA_MAX=2 TOKENIZERS_PARALLELISM=false \
  setsid nohup python train/rollout.py >> "$LOGD/rollout.log" 2>&1 < /dev/null &
echo $! > "$LOGD/rollout.pid"; sleep 25
echo "workers now: $(pgrep -c AgentFlow) | trainer alive: $(pgrep -f 'agentflow.ver[l]' | wc -l) | load: $(cut -d' ' -f1 /proc/loadavg)"
