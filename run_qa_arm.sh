#!/bin/bash
# Generalised QA-task arm runner: 2wiki | musique | hotpotqa | bamboogle.
# Supersedes run_2wiki_arm.sh (same semantics, task is now a parameter).
#
#   arm=agentflow : planner_main = AgentFlow/agentflow-planner-7b (:8000, trained planner)
#   arm=base      : planner_main = Qwen/Qwen2.5-7B-Instruct      (:8001, executor's base model)
#
# Helpers (planner_fixed / verifier / executor / tool engines) are Qwen2.5-7B-Instruct on
# :8001 in both arms. Toolbox: Base_Generator + Python_Coder + Wikipedia (Google is paid and
# stays out). Only planner_main differs between arms.
#
# Usage: ./run_qa_arm.sh <task> <agentflow|base> [N | "0,5,11"] [THREADS] [SUFFIX]
set -u
cd /data/sakhaei/code/AgentFlow
source .venv/bin/activate
export HF_HOME=/data/sakhaei/hf
export VLLM_BASE_URL="http://localhost:8001/v1"
export VLLM_API_KEY="dummy-token"
export PYTHONPATH=/data/sakhaei/code/AgentFlow
unset OPENAI_API_KEY GOOGLE_API_KEY DASHSCOPE_API_KEY GEMINI_API_KEY ANTHROPIC_API_KEY

TASK=${1:?task required: 2wiki|musique|hotpotqa|bamboogle|gameof24|aime24}
ARM=${2:?arm required: agentflow|base|qwen32b|qwen72b}
ARG=${3:-100}
THREADS=${4:-4}
SUFFIX=${5:-}
# PATCHED (MAReasoning) 2026-09-23: helper model/URL overridable so a Qwen3-8B base-vs-trained
# comparison can run with Qwen3-8B helpers (HELPER_MODEL=vllm-Qwen/Qwen3-8B HELPER_URL=http://host:port/v1).
EXEC="${HELPER_MODEL:-vllm-Qwen/Qwen2.5-7B-Instruct}"
export VLLM_BASE_URL="${HELPER_URL:-$VLLM_BASE_URL}"

case "$ARM" in
  agentflow) LLM="vllm-AgentFlow/agentflow-planner-7b"; PURL="http://localhost:8000/v1" ;;
  base)      LLM="vllm-Qwen/Qwen2.5-7B-Instruct";       PURL="http://localhost:8001/v1" ;;
  # 32B planner lives on mds2's H100; helpers stay on mds1:8001 so only planner_main changes
  qwen32b)   LLM="vllm-Qwen/Qwen2.5-32B-Instruct-AWQ"; PURL="http://mds1srv2.epfl.ch:8000/v1" ;;
  # 32B and 72B time-share mds2's single H100, so only one of these arms can run at a time
  qwen72b)   LLM="vllm-Qwen/Qwen2.5-72B-Instruct-AWQ"; PURL="http://mds1srv2.epfl.ch:8000/v1" ;;
  # PATCHED (MAReasoning) 2026-09-23: generic arm  custom:<served-model-name>:<base-url>  for any
  # vLLM endpoint (e.g. a merged Flow-GRPO checkpoint); label uses the model basename.
  custom:*)  LLM="vllm-$(echo "$ARM" | cut -d: -f2)"; PURL="$(echo "$ARM" | cut -d: -f3-)" ;;
  *) echo "unknown arm: $ARM"; exit 1 ;;
esac
LABEL="$TASK-$(echo "$ARM" | sed -E "s#^custom:([^:]*):.*#\\1#; s#/#_#g")$SUFFIX"
# PATCHED (MAReasoning) 2026-09-18: bamboogle arms must match the AgentFlow-7B baseline
# (run_bamboogle_local.sh): Base_Generator + Wikipedia only, no Python_Coder.
# PATCHED (MAReasoning) 2026-09-28: math tasks (gameof24, aime24) get Base_Generator + Python_Coder as in
# run_gameof24_local.sh; scoring picks score_gameof24.py for gameof24 (expression check), score_qa.py otherwise.
case "$TASK" in
  bamboogle)        TOOLS="Base_Generator_Tool,Wikipedia_Search_Tool"; TENG="$EXEC,$EXEC" ;;
  gameof24|aime24|amc23) TOOLS="Base_Generator_Tool,Python_Coder_Tool"; TENG="$EXEC,$EXEC" ;;
  *)                TOOLS="Base_Generator_Tool,Python_Coder_Tool,Wikipedia_Search_Tool"; TENG="$EXEC,$EXEC,$EXEC" ;;
esac
SCORER=score_qa.py; [ "$TASK" = "gameof24" ] && SCORER=score_gameof24.py

if [[ "$ARG" == *","* ]]; then INDICES=$(echo "$ARG" | tr ',' '\n'); else INDICES=$(seq 0 $((ARG-1))); fi

echo "=== task=$TASK arm=$ARM planner=$LLM label=$LABEL ==="

cd test
OUT="$TASK/results/$LABEL"; LOG="$TASK/logs/$LABEL"; CACHE="$TASK/cache-$LABEL"
mkdir -p "$OUT" "$LOG" "$CACHE"

run_one() {
  i=$1
  [ -f "$OUT/output_$i.json" ] && return 0
  python solve.py --index "$i" --task "$TASK" --data_file "$TASK/data/data.json" \
    --llm_engine_name "$LLM" --root_cache_dir "$CACHE" --output_json_dir "$OUT" \
    --output_types direct \
    --enabled_tools "$TOOLS" \
    --tool_engine "$TENG" \
    --model_engine "trainable,$EXEC,$EXEC,$EXEC" \
    --max_time 300 --max_steps 10 --temperature 0.0 \
    --base_url "$PURL" > "$LOG/$i.log" 2>&1
  echo "done $i"
}
export -f run_one; export OUT LOG CACHE TASK LLM EXEC PURL TOOLS TENG

echo "$INDICES" | xargs -P "$THREADS" -I{} bash -c "run_one {}"
echo "=== $LABEL finished ==="
python /data/sakhaei/code/AgentFlow/$SCORER \
  --data_file "$TASK/data/data.json" --result_dir "$OUT" | tee "$OUT/finalscore.log"
