#!/bin/bash
# PATCHED (MAReasoning) 2026-09-24. Supervisor: relaunch the run after a crash until it finishes.
# verl resumes from the latest checkpoint in trainer.default_local_dir (resume_mode auto), so a
# crash costs at most save_freq steps. Judge run 4 died at step 21 with an optimizer-step OOM after
# 20 clean steps; rather than lose a night, restart automatically.
# Usage: train/keep_training.sh <config.yaml> <tag-base> [max_restarts]
set -u
cd /data/sakhaei/code/AgentFlow
CFG=${1:?config}; BASE=${2:?tag base}; MAX=${3:-8}
CKPT_DIR=/data/sakhaei/ckpt/AgentFlow_mds/$(python3 -c "import yaml,sys; print(yaml.safe_load(open('$CFG'))['env']['EXPERIMENT_NAME'])")
for i in $(seq 1 "$MAX"); do
  TAG="${BASE}_$(date +%m%d%H%M)_r$i"; L=/data/sakhaei/runs/train/$TAG   # timestamped: a second supervisor instance must not clobber an earlier attempt's log
  train/stop_mds.sh >/dev/null 2>&1
  # 2026-09-29: r2 died of host OOM 40 min after a restart -- the previous trainer's offloaded memory had not
  # been released yet. Wait for host RAM to drop below 120 GB (up to 15 min) before relaunching.
  for w in $(seq 1 90); do [ "$(free -g | awk 'NR==2{print $3}')" -lt 120 ] && break; sleep 10; done
  echo "$(date '+%F %T') host RAM before relaunch: $(free -g | awk 'NR==2{print $3}') GB"
  train/launch_mds.sh "$CFG" "$TAG" | tail -1
  sleep 120
  while pgrep -f "agentflow.ver[l]" >/dev/null; do sleep 60; done
  if grep -q "Training finished" "$L/train.log" 2>/dev/null; then
    echo "$(date '+%F %T') finished in $TAG"; exit 0
  fi
  echo "$(date '+%F %T') trainer exited without finishing ($TAG); last ckpt: $(cat "$CKPT_DIR/latest_checkpointed_iteration.txt" 2>/dev/null); restarting"
done
echo "$(date '+%F %T') gave up after $MAX restarts"
