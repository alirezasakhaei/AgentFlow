#!/bin/bash
# PATCHED (MAReasoning) 2026-09-29. Rollout workers leak host memory (one AgentFlow-Worker reached 41 GB
# RSS, all workers 34 -> 74 GB in 5 h); the node then hits Ray's kill threshold. When the workers' total RSS
# exceeds LIMIT_GB, restart only the workers right after the next "step:N" line (= start of a new
# generation phase, so almost no in-flight rollouts are lost). Usage: worker_watchdog.sh <config> <tag-glob>
set -u
CFG=${1:?config}; GLOB=${2:?tag glob}; LIMIT_GB=${3:-90}
cd /data/sakhaei/code/AgentFlow
log(){ echo "$(date "+%F %T") $*"; }
wrss(){ ps -eo rss,args | awk "/rollout.py|AgentFlow-|spawn_main/ && !/awk/ {s+=\$1} END {printf \"%d\", s/1048576}"; }
while true; do
  r=$(wrss)
  if [ "$r" -gt "$LIMIT_GB" ]; then
    L=$(ls -t /data/sakhaei/runs/train/${GLOB}/train.log 2>/dev/null | head -1); TAG=$(basename "$(dirname "$L")")
    n0=$(grep -c "step:[0-9]* - agent" "$L")
    log "workers at ${r} GB > ${LIMIT_GB}; waiting for the next step boundary of $TAG"
    for i in $(seq 1 240); do sleep 30; [ "$(grep -c "step:[0-9]* - agent" "$L")" -gt "$n0" ] && break; done
    L=$(ls -t /data/sakhaei/runs/train/${GLOB}/train.log 2>/dev/null | head -1); TAG=$(basename "$(dirname "$L")")   # PATCHED (MAReasoning): re-resolve, the run may have restarted while we waited
    log "restarting workers ($TAG)"; train/restart_workers.sh "$CFG" "$TAG" 2>&1 | tail -1
    sleep 600
  fi
  sleep 120
done
