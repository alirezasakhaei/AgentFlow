#!/bin/bash
# MAReasoning container entrypoint (2026-09-30). Modes:
#   train <config.yaml> [tag]   ray head (all visible GPUs) + rollout workers + verl trainer, FOREGROUND; exits with the trainer's code
#   serve <model> [vllm args]   vllm serve (helper / judge / eval planner)
#   eval  <args>                the eval harness (eval_one_arm.sh args)
#   bash | <any command>        passthrough
# Env: AF_RUNS (logs/pids root, default /data/runs), N_GPUS (default = visible GPUs), and every key of the
# config's env: block is exported UNLESS already set in the environment, so `docker run -e` / `runai --environment` win.
set -u
cd /workspace/AgentFlow
mode=${1:-bash}; shift || true
case "$mode" in
train)
  export AGENTFLOW_TRAIN_CONFIG=${1:?config yaml}; TAG=${2:-$(date +%Y%m%d-%H%M%S)}
  LOGD=${AF_RUNS:-/data/runs}/train/$TAG; mkdir -p "$LOGD"
  export HYDRA_FULL_ERROR=1 RAY_memory_usage_threshold=${RAY_memory_usage_threshold:-0.97}
  unset OPENAI_API_KEY GOOGLE_API_KEY DASHSCOPE_API_KEY GEMINI_API_KEY ANTHROPIC_API_KEY
  eval "$(python - <<PY
import yaml, os, shlex
for k, v in yaml.safe_load(open(os.environ["AGENTFLOW_TRAIN_CONFIG"]))["env"].items():
    if k not in os.environ: print(f"export {k}={shlex.quote(str(v))}")
PY
)"
  if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] || [ "${CUDA_VISIBLE_DEVICES}" = "all" ]; then
    export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,); fi
  export N_GPUS=${N_GPUS:-$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)}
  echo "image=${AF_IMAGE_GIT_SHA:-?} config=$AGENTFLOW_TRAIN_CONFIG tag=$TAG gpus=$CUDA_VISIBLE_DEVICES n=$N_GPUS logs=$LOGD"
  ray stop --force >/dev/null 2>&1 || true
  ray start --head --num-gpus "$N_GPUS" --object-store-memory ${RAY_OBJECT_STORE_BYTES:-40000000000} --dashboard-host=127.0.0.1 > "$LOGD/ray.log" 2>&1
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MALLOC_ARENA_MAX=2 TOKENIZERS_PARALLELISM=false \
    setsid python train/rollout.py > "$LOGD/rollout.log" 2>&1 < /dev/null & echo $! > "$LOGD/rollout.pid"
  sleep 5
  python train/train_agent.py 2>&1 | tee "$LOGD/train.log"; rc=${PIPESTATUS[0]}
  echo "trainer exited rc=$rc"; pkill -TERM -s "$(ps -o sid= -p "$(cat "$LOGD/rollout.pid")" | tr -d ' ')" 2>/dev/null
  ray stop --force >/dev/null 2>&1; exit "$rc" ;;
serve)  exec vllm serve "$@" ;;
eval)   exec ./eval_one_arm.sh "$@" ;;
bash)   exec bash "$@" ;;
*)      exec "$mode" "$@" ;;
esac
