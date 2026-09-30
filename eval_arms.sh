#!/bin/bash
# PATCHED (MAReasoning) 2026-09-28. Paired eval: base Qwen3-8B planner vs Flow-GRPO step-100 planner,
# same harness, same frozen Qwen3-8B helpers on mds2, rule-based scoring, helper tokens metered.
# Usage: eval_arms.sh [tag]   (needs serve_eval_planners.sh up: :8000 trained, :8001 base)
set -u
cd /data/sakhaei/code/AgentFlow
export HELPER_MODEL=vllm-Qwen/Qwen3-8B-helper HELPER_URL=http://mds1srv2.epfl.ch:8000/v1
export AGENTFLOW_DISABLE_THINKING=1 HF_HOME=/data/sakhaei/hf
R=/data/sakhaei/runs; TAG=${1:-eval100}; mkdir -p $R/$TAG
run_arm() {  # $1 arm spec  $2 short name
  local ARM=$1 NAME=$2
  source .venv/bin/activate
  python token_meter.py snap $R/$TAG/${NAME}_before.json
  ./run_qa_arm.sh bamboogle "$ARM" 125 6 > $R/$TAG/${NAME}_bamboogle.log 2>&1
  ./run_qa_arm.sh 2wiki     "$ARM" 50  6 > $R/$TAG/${NAME}_2wiki.log 2>&1
  ./run_qa_arm.sh gameof24  "$ARM" 30  6 > $R/$TAG/${NAME}_gameof24.log 2>&1
  ./run_qa_arm.sh aime24    "$ARM" 30  6 > $R/$TAG/${NAME}_aime24.log 2>&1
  python token_meter.py snap $R/$TAG/${NAME}_after.json
  echo "$(date '+%F %T') arm $NAME done"
}
# The two arms run concurrently: each has its own planner GPU; the helper on mds2 serves both, as it
# served 32 training workers. Token meters are per-arm snapshots of the SAME counters, so the shared
# helper counter is attributed by difference only when arms run alone -- record wall-clock, and use
# the per-arm request logs (proxy) if a split is needed. (Kept simple: helper tokens reported jointly.)
run_arm "custom:Qwen/Qwen3-8B:http://localhost:8001/v1"        base    &
run_arm "custom:Qwen3-8B-grpo-step100:http://localhost:8000/v1" trained &
wait
echo "$(date '+%F %T') ######## EVAL DONE ########"
