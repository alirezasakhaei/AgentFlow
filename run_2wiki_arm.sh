#!/bin/bash
# 2wiki, two planner arms over an IDENTICAL executor / tool stack.
#
#   arm=agentflow : planner_main = AgentFlow/agentflow-planner-7b (:8000, the trained planner)
#   arm=base      : planner_main = Qwen/Qwen2.5-7B-Instruct      (:8001, the executor's own
#                   base model reused as planner -- upstream's "Qwen2.5-7b-naive" baseline)
#
# Everything else is held fixed across arms: the same helper pod on :8001 serves
# planner_fixed / verifier / executor / all tool engines, the same three-tool toolbox, the
# same decoding settings, the same item indices. Only planner_main changes.
#
# NOTE: executor.py::split_commands is left UNPATCHED on purpose. It silently drops any
# command that is not a single `execution = tool.execute(...)` binding. That bug hits both
# arms equally, so the arm comparison stays valid; only the absolute rates are depressed.
#
# Usage: ./run_2wiki_arm.sh <agentflow|base> [N | "0,5,11"] [THREADS]
set -u
cd /data/sakhaei/code/AgentFlow
source .venv/bin/activate
export HF_HOME=/data/sakhaei/hf
export VLLM_BASE_URL="http://localhost:8001/v1"   # helpers + tool engines, both arms
export VLLM_API_KEY="dummy-token"
export PYTHONPATH=/data/sakhaei/code/AgentFlow
unset OPENAI_API_KEY GOOGLE_API_KEY DASHSCOPE_API_KEY GEMINI_API_KEY ANTHROPIC_API_KEY

ARM=${1:?arm required: agentflow|base}
ARG=${2:-50}
THREADS=${3:-4}
SUFFIX=${4:-}          # e.g. "-fix" to keep patched-harness results in their own label
TASK=2wiki
EXEC="vllm-Qwen/Qwen2.5-7B-Instruct"

case "$ARM" in
  agentflow) LLM="vllm-AgentFlow/agentflow-planner-7b"; PURL="http://localhost:8000/v1"; LABEL="arm-agentflow$SUFFIX" ;;
  base)      LLM="vllm-Qwen/Qwen2.5-7B-Instruct";       PURL="http://localhost:8001/v1"; LABEL="arm-base$SUFFIX" ;;
  *) echo "unknown arm: $ARM"; exit 1 ;;
esac

if [[ "$ARG" == *","* ]]; then INDICES=$(echo "$ARG" | tr ',' '\n'); else INDICES=$(seq 0 $((ARG-1))); fi

echo "=== arm=$ARM  planner=$LLM  url=$PURL  label=$LABEL ==="

cd test
OUT="$TASK/results/$LABEL"; LOG="$TASK/logs/$LABEL"; CACHE="$TASK/cache-$LABEL"
mkdir -p "$OUT" "$LOG" "$CACHE"

run_one() {
  i=$1
  [ -f "$OUT/output_$i.json" ] && return 0
  python solve.py --index "$i" --task "$TASK" --data_file "$TASK/data/data.json" \
    --llm_engine_name "$LLM" --root_cache_dir "$CACHE" --output_json_dir "$OUT" \
    --output_types direct \
    --enabled_tools "Base_Generator_Tool,Python_Coder_Tool,Wikipedia_Search_Tool" \
    --tool_engine "$EXEC,$EXEC,$EXEC" \
    --model_engine "trainable,$EXEC,$EXEC,$EXEC" \
    --max_time 300 --max_steps 10 --temperature 0.0 \
    --base_url "$PURL" > "$LOG/$i.log" 2>&1
  echo "done $i"
}
export -f run_one; export OUT LOG CACHE TASK LLM EXEC PURL

echo "$INDICES" | xargs -P "$THREADS" -I{} bash -c "run_one {}"
echo "=== arm $ARM finished ==="
