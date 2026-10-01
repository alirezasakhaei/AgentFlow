#!/bin/bash
# MAReasoning 2026-10-01: stop whatever training is running (Exp 3 resume), launch Exp 4 (rule reward) with supervisor + watchdog.
set -u
K=keep_train; W=worker_watch
cd /data/sakhaei/code/AgentFlow
log(){ echo "$(date '+%F %T') $*"; }
log "stopping current training"
pkill -f "${K}in[g]"; pkill -f "${W}do[g]"; train/stop_mds.sh 2>&1 | tail -1; sleep 5
log "GPU: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | paste -s) RAM used: $(free -g | awk 'NR==2{print $3}') GB"
until [ "$(free -g | awk 'NR==2{print $3}')" -lt 120 ]; do sleep 20; done
curl -sf -m 5 -H "Authorization: Bearer dummy-token" http://mds1srv2.epfl.ch:8000/v1/models | grep -q "Qwen3-8B-helper" || { log "helper on mds2 not up"; exit 1; }
(setsid nohup train/${K}ing.sh train/config_mds_exp4.yaml flowgrpo_qwen3_8b_rule_lr3e6pen 20 > /data/sakhaei/runs/train/${K}ing_exp4.log 2>&1 < /dev/null &)
sleep 90
(setsid nohup train/${W}dog.sh train/config_mds_exp4.yaml "flowgrpo_qwen3_8b_rule_lr3e6pen*" 90 >> /data/sakhaei/runs/train/${W}dog.log 2>&1 < /dev/null &)
sleep 5; log "exp 4 launched: supervisor $(pgrep -f "${K}in[g]" | wc -l) watchdog $(pgrep -f "${W}do[g]" | wc -l); $(tail -1 /data/sakhaei/runs/train/${K}ing_exp4.log)"
log "######## EXP4 STARTED ########"
