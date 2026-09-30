#!/bin/bash
# 2wiki on the local two-server topology, Wikipedia-only search + the Python coder.
#   :8000 AgentFlow/agentflow-planner-7b  -> planner_main (the model under test)
#   :8001 Qwen/Qwen2.5-7B-Instruct        -> planner_fixed, verifier, executor, tool engines
#
# Differs from run_bamboogle_local.sh in one deliberate way: Python_Coder_Tool IS enabled.
# 2wiki's comparison items are retrieve-two-facts-then-compare, and the open question is
# whether the planner delegates that comparison to the coder or does it in its own head.
# With the coder absent the question is unaskable. Google_Search_Tool stays out (paid key).
#
# Usage: ./run_2wiki_local.sh [N | "0,5,11"] [THREADS] [LABEL]
set -u
cd /data/sakhaei/code/AgentFlow
source .venv/bin/activate
export HF_HOME=/data/sakhaei/hf
export VLLM_BASE_URL="http://localhost:8001/v1"   # everything that is not the planner
export VLLM_API_KEY="dummy-token"
export PYTHONPATH=/data/sakhaei/code/AgentFlow

# Hard guarantee that no paid backend is reachable even if a key is lying around.
unset OPENAI_API_KEY GOOGLE_API_KEY DASHSCOPE_API_KEY GEMINI_API_KEY ANTHROPIC_API_KEY

TASK=2wiki
ARG=${1:-10}
THREADS=${2:-4}
LABEL=${3:-AgentFlow-7B-localqwen-2wiki}
LLM="vllm-AgentFlow/agentflow-planner-7b"
EXEC="vllm-Qwen/Qwen2.5-7B-Instruct"

# first arg is either a count (run 0..N-1) or an explicit comma-separated index list
if [[ "$ARG" == *","* ]]; then
  INDICES=$(echo "$ARG" | tr ',' '\n')
else
  INDICES=$(seq 0 $((ARG-1)))
fi

cd test
OUT="$TASK/results/$LABEL"; LOG="$TASK/logs/$LABEL"; CACHE="$TASK/cache"
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
    --base_url "http://localhost:8000/v1" > "$LOG/$i.log" 2>&1
  echo "done $i"
}
export -f run_one; export OUT LOG CACHE TASK LLM EXEC

echo "$INDICES" | xargs -P "$THREADS" -I{} bash -c "run_one {}"

python /data/sakhaei/code/AgentFlow/score_qa.py \
  --data_file "$TASK/data/data.json" --result_dir "$OUT" | tee "$OUT/finalscore.log"
