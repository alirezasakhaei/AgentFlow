#!/bin/bash
# Supervisor for the upstream clone (same loop as train/keep_training.sh in the fork). Usage: keep_training_upstream.sh <config.yaml> <tag-base> [max_restarts]
set -u
CFG=${1:?config}; BASE=${2:?tag base}; MAX=${3:-8}
U=/data/sakhaei/code/AgentFlow-upstream
CKPT_DIR=/data/sakhaei/ckpt/AgentFlow_mds/$(python3 -c "import yaml; print(yaml.safe_load(open('$CFG'))['env']['EXPERIMENT_NAME'])")
log(){ echo "$(date '+%F %T') $*"; }
for i in $(seq 1 $MAX); do
  timeout 120 $U/.venv/bin/ray stop --force >/dev/null 2>&1   # the upstream venv owns this Ray cluster; the fork venv ray (other version) hangs on it
  timeout 300 /data/sakhaei/code/AgentFlow/train/stop_mds.sh >/dev/null 2>&1
  until [ "$(free -g | awk 'NR==2{print $3}')" -lt 120 ]; do sleep 20; done
  log "host RAM before relaunch: $(free -g | awk 'NR==2{print $3}') GB"
  TAG=${BASE}_$(date +%m%d%H%M)_r$i
  /data/sakhaei/runs/launch_upstream.sh "$CFG" "$TAG" | tail -1
  sleep 60
  while kill -0 "$(cat /data/sakhaei/runs/train/$TAG/train.pid)" 2>/dev/null; do sleep 60; done
  if grep -q "Training finished\|Final validation" /data/sakhaei/runs/train/$TAG/train.log 2>/dev/null; then log "training finished ($TAG)"; break; fi
  log "trainer exited without finishing ($TAG); last ckpt: $(cat $CKPT_DIR/latest_checkpointed_iteration.txt 2>/dev/null); restarting"
done
