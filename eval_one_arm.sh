#!/bin/bash
# PATCHED (MAReasoning) 2026-09-28. One planner arm over the four eval tasks (same sequence as eval_arms.sh).
# Usage: eval_one_arm.sh <custom:name:url> <short-name> [tag]
set -u
cd /data/sakhaei/code/AgentFlow
export HELPER_MODEL=vllm-Qwen/Qwen3-8B-helper HELPER_URL=http://mds1srv2.epfl.ch:8000/v1
export AGENTFLOW_DISABLE_THINKING=1 HF_HOME=/data/sakhaei/hf
ARM=$1; NAME=$2; TAG=${3:-eval100}; R=/data/sakhaei/runs/$TAG; mkdir -p $R
source .venv/bin/activate
python token_meter.py snap $R/${NAME}_before.json
./run_qa_arm.sh bamboogle "$ARM" 125 6 > $R/${NAME}_bamboogle.log 2>&1
./run_qa_arm.sh 2wiki     "$ARM" 50  6 > $R/${NAME}_2wiki.log 2>&1
./run_qa_arm.sh gameof24  "$ARM" 30  6 > $R/${NAME}_gameof24.log 2>&1
./run_qa_arm.sh aime24    "$ARM" 30  6 > $R/${NAME}_aime24.log 2>&1
python token_meter.py snap $R/${NAME}_after.json
echo "$(date "+%F %T") arm $NAME done"
