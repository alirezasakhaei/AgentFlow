#!/bin/bash
# bamboogle with a fully local two-server topology, Wikipedia-only search:
#   :8000 AgentFlow/agentflow-planner-7b  -> planner_main (the model under test)
#   :8001 Qwen/Qwen2.5-7B-Instruct        -> planner_fixed, verifier, executor, tool engines
# Search is Wikipedia-only: Google_Search_Tool (needs GOOGLE_API_KEY) is NOT enabled, and
# Web_RAG_Search_Tool no longer leaks into the toolbox (see PATCHED notes in
# tools/wikipedia_search/tool.py). Chunk ranking uses a local CPU bge-small encoder, so
# there is no paid call anywhere in this run.
#
# Usage: ./run_bamboogle_local.sh [N] [THREADS] [LABEL]      (N defaults to all 125)
set -u
cd /data/sakhaei/code/AgentFlow
source .venv/bin/activate
export HF_HOME=/data/sakhaei/hf
export VLLM_BASE_URL="http://localhost:8001/v1"   # everything that is not the planner
export VLLM_API_KEY="dummy-token"
export PYTHONPATH=/data/sakhaei/code/AgentFlow

# Hard guarantee that no paid backend can be reached even if a key is lying around.
unset OPENAI_API_KEY GOOGLE_API_KEY DASHSCOPE_API_KEY GEMINI_API_KEY ANTHROPIC_API_KEY

TASK=bamboogle
N=${1:-125}
THREADS=${2:-8}
LABEL=${3:-AgentFlow-7B-localqwen-wiki}
LLM="vllm-AgentFlow/agentflow-planner-7b"
EXEC="vllm-Qwen/Qwen2.5-7B-Instruct"

cd test
OUT="$TASK/results/$LABEL"; LOG="$TASK/logs/$LABEL"; CACHE="$TASK/cache"
mkdir -p "$OUT" "$LOG" "$CACHE"

run_one() {
  i=$1
  [ -f "$OUT/output_$i.json" ] && return 0
  python solve.py --index "$i" --task "$TASK" --data_file "$TASK/data/data.json" \
    --llm_engine_name "$LLM" --root_cache_dir "$CACHE" --output_json_dir "$OUT" \
    --output_types direct \
    --enabled_tools "Base_Generator_Tool,Wikipedia_Search_Tool" \
    --tool_engine "$EXEC,$EXEC" \
    --model_engine "trainable,$EXEC,$EXEC,$EXEC" \
    --max_time 300 --max_steps 10 --temperature 0.0 \
    --base_url "http://localhost:8000/v1" > "$LOG/$i.log" 2>&1
  echo "done $i"
}
export -f run_one; export OUT LOG CACHE TASK LLM EXEC

seq 0 $((N-1)) | xargs -P "$THREADS" -I{} bash -c "run_one {}"

python /data/sakhaei/code/AgentFlow/score_bamboogle.py \
  --data_file "$TASK/data/data.json" --result_dir "$OUT" | tee "$OUT/finalscore.log"
