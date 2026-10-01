#!/bin/bash
# MAReasoning 2026-10-01: evaluate a checkpoint of the RUNNING experiment without pausing it.
#   merge on mds1 (CPU) -> rsync HF to mds2 -> serve on mds2:8002 beside the half-card helper -> 4 tasks from mds1 -> compare -> kill server
# Usage: eval_ckpt_live.sh <exp-name> <step> <short-tag>      e.g. eval_ckpt_live.sh flowgrpo_qwen3_8b_rule_lr3e6_pen 15 exp4
set -u
EXP=${1:?exp}; STEP=${2:?step}; SHORT=${3:?short}
CK=/data/sakhaei/ckpt/AgentFlow_mds/$EXP/global_step_$STEP
HF=/data/sakhaei/ckpt/hf/${EXP}_step$STEP
NAME=Qwen3-8B-${SHORT}-step$STEP           # no gpt/o1/o3/o4 substrings
TAG=eval_${SHORT}_s$STEP; R=/data/sakhaei/runs/$TAG; mkdir -p $R
URL=http://mds1srv2.epfl.ch:8002/v1
cd /data/sakhaei/code/AgentFlow
log(){ echo "$(date '+%F %T') $*"; }
[ -f $CK/actor/model_world_size_2_rank_1.pt ] || { log "no checkpoint $CK"; exit 1; }
if [ ! -f $HF/config.json ]; then
  log "merging step $STEP (CPU)"; source .venv-train/bin/activate
  HF_HOME=/data/sakhaei/hf CUDA_VISIBLE_DEVICES= nice -n 10 python -m verl.model_merger merge --backend fsdp --local_dir $CK/actor --target_dir $HF > $R/merge.log 2>&1 || { log "merge failed"; tail -3 $R/merge.log; exit 1; }
  deactivate; log "merged $(du -sh $HF | cut -f1)"
fi
log "rsync to mds2"; rsync -a --info=progress2 $HF/ mds2:$HF/ > $R/rsync.log 2>&1 || { log "rsync failed"; tail -2 $R/rsync.log; exit 1; }
log "serving on mds2:8002 as $NAME"
ssh mds2 "V=vllm; pkill -f \"\$V serve .*--port 800[2]\" 2>/dev/null; sleep 3; (setsid nohup /data/sakhaei/runs/serve_eval_planner.sh $HF $NAME 8002 > /data/sakhaei/runs/serve_eval_$NAME.log 2>&1 < /dev/null &)"   # V=... : the pattern must not appear literally in the remote shell's own cmdline (self-kill)
for i in $(seq 1 90); do sleep 5; curl -sf -m 3 -H "Authorization: Bearer dummy-token" $URL/models 2>/dev/null | grep -q "$NAME" && { log "planner up after $((i*5))s"; break; }; [ $i = 90 ] && { log "planner not up"; ssh mds2 "tail -3 /data/sakhaei/runs/serve_eval_$NAME.log"; exit 1; }; done
export HELPER_MODEL=vllm-Qwen/Qwen3-8B-helper HELPER_URL=http://mds1srv2.epfl.ch:8000/v1 AGENTFLOW_DISABLE_THINKING=1 HF_HOME=/data/sakhaei/hf
source .venv/bin/activate
python token_meter.py snap $R/before.json
log "running 4 tasks (8 threads) against $URL"
./run_qa_arm.sh bamboogle "custom:$NAME:$URL" 125 8 > $R/bamboogle.log 2>&1
./run_qa_arm.sh 2wiki     "custom:$NAME:$URL" 50  8 > $R/2wiki.log 2>&1
./run_qa_arm.sh gameof24  "custom:$NAME:$URL" 30  8 > $R/gameof24.log 2>&1
./run_qa_arm.sh aime24    "custom:$NAME:$URL" 30  8 > $R/aime24.log 2>&1
python token_meter.py snap $R/after.json
log "eval done:"
{ echo "### $NAME vs base (Qwen_Qwen3-8B, 2026-09-29)"; python compare_two_arms.py Qwen_Qwen3-8B $NAME; } | tee $R/compare.txt
deactivate
ssh mds2 "V=vllm; pkill -f \"\$V serve .*--port 800[2]\""; sleep 3; log "eval server stopped ($(ssh mds2 "pgrep -fc \"vllm serv[e]\"") vllm left on mds2)"
log "######## LIVE EVAL $NAME DONE ########"
