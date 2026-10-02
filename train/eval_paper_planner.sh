#!/bin/bash
# MAReasoning 2026-10-02: the paper's released planner (AgentFlow/agentflow-planner-7b) vs its own base (Qwen2.5-7B-Instruct),
# both with Qwen2.5-7B-Instruct as every frozen helper (the planner's native setup), all four tasks, same session, same harness.
# GPU0 :8000 released planner; GPU1 :8001 Qwen2.5-7B-Instruct = base-arm planner AND helpers for both arms.
set -u
K=keep_train; W=worker_watch
TAG=eval_paper_planner; R=/data/sakhaei/runs/$TAG; mkdir -p $R
cd /data/sakhaei/code/AgentFlow
log(){ echo "$(date '+%F %T') $*"; }
export HF_HOME=/data/sakhaei/hf VLLM_API_KEY=dummy-token VLLM_USE_V1=1
if pgrep -f "vll[m] serve" >/dev/null; then log "vLLM servers already present, not restarting"; else
log "stopping Exp 4"; pkill -f "${K}in[g]"; pkill -f "${W}do[g]"; train/stop_mds.sh 2>&1 | tail -1; sleep 8
nvidia-smi --query-gpu=memory.used --format=csv,noheader | paste -s
source .venv-train/bin/activate
CUDA_VISIBLE_DEVICES=0 setsid nohup .venv-train/bin/vllm serve AgentFlow/agentflow-planner-7b --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.85 \
  --max-model-len 16384 --max-num-seqs 64 --served-model-name AgentFlow/agentflow-planner-7b > $R/serve_planner.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=1 setsid nohup .venv-train/bin/vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8001 --gpu-memory-utilization 0.85 \
  --max-model-len 16384 --max-num-seqs 128 --served-model-name Qwen/Qwen2.5-7B-Instruct > $R/serve_base.log 2>&1 < /dev/null &
deactivate
fi
for i in $(seq 1 240); do sleep 5; n=0   # cold disk: two 7B loads took ~7 min
  curl -sf -m 3 -H "Authorization: Bearer dummy-token" http://localhost:8000/v1/models 2>/dev/null | grep -q "planner-7b" && n=$((n+1))
  curl -sf -m 3 -H "Authorization: Bearer dummy-token" http://localhost:8001/v1/models 2>/dev/null | grep -q "Qwen2.5-7B-Instruct" && n=$((n+1))
  [ $n = 2 ] && { log "both servers up after $((i*5))s"; break; }
done
[ "${n:-0}" = 2 ] || { log "servers not up"; tail -2 $R/serve_planner.log $R/serve_base.log | cut -c1-160; exit 1; }
export HELPER_MODEL=vllm-Qwen/Qwen2.5-7B-Instruct HELPER_URL=http://localhost:8001/v1
unset AGENTFLOW_DISABLE_THINKING
source .venv/bin/activate
python token_meter.py snap $R/before.json
log "running both arms (8 threads each)"
( for t in bamboogle:125 2wiki:50 gameof24:30 aime24:30; do ./run_qa_arm.sh ${t%%:*} "custom:AgentFlow/agentflow-planner-7b:http://localhost:8000/v1" ${t##*:} 8 > $R/trained_${t%%:*}.log 2>&1; done ) &
P1=$!
( for t in bamboogle:125 2wiki:50 gameof24:30 aime24:30; do ./run_qa_arm.sh ${t%%:*} "custom:Qwen/Qwen2.5-7B-Instruct:http://localhost:8001/v1" ${t##*:} 8 > $R/base_${t%%:*}.log 2>&1; done ) &
P2=$!
wait $P1 $P2
python token_meter.py snap $R/after.json
log "eval done:"
{ echo "### paper planner (agentflow-planner-7b) vs its base (Qwen2.5-7B-Instruct), Qwen2.5-7B-Instruct helpers"; python compare_two_arms.py Qwen_Qwen2.5-7B-Instruct AgentFlow_agentflow-planner-7b; } | tee $R/compare.txt
deactivate
V=vllm; pkill -f "$V serv[e]"; log "servers stopped"
log "######## PAPER PLANNER EVAL DONE ########"
