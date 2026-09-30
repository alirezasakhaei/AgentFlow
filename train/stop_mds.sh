#!/bin/bash
# PATCHED (MAReasoning). Stops a training run started by launch_mds.sh: trainer, rollout workers, ray.
# Kills by SESSION id: launch_mds.sh starts rollout.py and train_agent.py with setsid, and the
# rollout workers are multiprocessing children whose cmdline is "python -c ...spawn_main..." (not
# train/rollout.py), so a name-based pkill left them alive; stale workers then polled the NEXT run's
# task server and posted junk rollouts into it (seen on smoke4 -> smoke5, 2026-09-23).
set -u
cd /data/sakhaei/code/AgentFlow; source .venv-train/bin/activate
for d in /data/sakhaei/runs/train/*/; do
  for f in rollout.pid train.pid; do
    [ -f "$d$f" ] || continue
    pid=$(cat "$d$f"); sid=$(ps -o sid= -p "$pid" 2>/dev/null | tr -d " ")
    [ -n "$sid" ] && { pkill -TERM -s "$sid" 2>/dev/null; echo "stopped session $sid ($d$f)"; }
  done
done
# workers rename themselves via setproctitle and outlive their session leader, so also kill
# whatever still holds a run's log files open (they inherit the stdout/stderr fds)
for d in /data/sakhaei/runs/train/*/; do fuser -k -TERM "$d/rollout.log" "$d/train.log" >/dev/null 2>&1; done
sleep 2
for d in /data/sakhaei/runs/train/*/; do fuser -k -KILL "$d/rollout.log" "$d/train.log" >/dev/null 2>&1; done
pkill -f "train/train_agent.py"; pkill -f "agentflow.verl"; pkill -f "train/rollout.py"
sleep 3
pkill -9 -f "train/rollout.py" 2>/dev/null; pkill -9 -f "agentflow.verl" 2>/dev/null
# any leftover spawn_main workers of THIS venv that are not ray's
for p in $(pgrep -f "multiprocessing.spawn"); do
  ps -o cmd= -p "$p" | grep -q "venv-train" && ! ps -o cmd= -p "$p" | grep -q "ray" && kill -9 "$p" 2>/dev/null
done
ray stop --force >/dev/null 2>&1 || true
sleep 2
echo "remaining venv-train python procs: $(pgrep -f "venv-train/bin/python" | wc -l)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
