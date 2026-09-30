# MAReasoning training image

One image for the whole Flow-GRPO pipeline: AgentFlow fork (our `PATCHED (MAReasoning)` edits), verl 0.5.0
(+3 patches), vLLM 0.9.2, torch 2.7.0 cu128, python 3.12. Exact package set: `requirements.lock`
(the validated mds1 venv). Runs as any uid.

## Build (mds1 or any docker host)
    docker/build.sh              # -> agentflow-train:<git sha>, :latest
    docker/build.sh push         # after: docker login registry.rcp.epfl.ch

## Volumes / paths
| in container | mds1 | RCP |
|---|---|---|
| `/data/hf` (HF_HOME) | `/data/sakhaei/hf` | `/nlpscratch/home/sakhaei/hf` |
| `/data/runs` (AF_RUNS: logs, pids) | `/data/sakhaei/runs` | `/nlpscratch/home/sakhaei/agentflow/runs` |
| `/data/ckpt` (trainer.default_local_dir) | `/data/sakhaei/ckpt` | `/nlpscratch/home/sakhaei/agentflow/ckpt` |
| `/workspace/AgentFlow/data` (train/val parquet) | `/data/sakhaei/code/AgentFlow/data` | PVC copy |
The mds configs still name `/data/sakhaei/...`; on mds1 bind-mount `/data/sakhaei:/data/sakhaei` and they work
unchanged. For RCP use a config whose paths point under `/data/...` (container paths above) and pass secrets /
URLs with `--environment` (config env keys already set in the environment are NOT overridden).

## Run
    # trainer, all visible GPUs (N_GPUS defaults to the visible count)
    docker run --rm --gpus all --shm-size 64g --ulimit memlock=-1 --network host \
      -v /data/sakhaei:/data/sakhaei -v /data/sakhaei/hf:/data/hf -v /data/sakhaei/runs:/data/runs \
      agentflow-train:latest train train/config_mds_exp3.yaml mytag
    # helper / judge / eval planner
    docker run --rm --gpus '"device=0"' --network host -v /data/sakhaei/hf:/data/hf agentflow-train:latest \
      serve Qwen/Qwen3-8B --served-model-name Qwen/Qwen3-8B-helper --port 8000 --max-model-len 16384
    # RCP (single pod, N GPUs): runai submit ... -i registry.rcp.epfl.ch/criteria-feedback/agentflow-train:<sha> --gpu N \
    #   --pvc nlp-scratch:/nlpscratch --large-shm --command -- /workspace/AgentFlow/docker/entrypoint.sh train <config> <tag>
Multi-GPU is the config's `N_GPUS`/`CUDA_VISIBLE_DEVICES` (Ray head sizes itself from it); the trainer, rollout
workers and (optionally) the helper vLLM can share one pod — put the helper on its own GPU and point
`VLLM_BASE_URL` at `http://localhost:8000/v1`. `--network host` matters on a bare host: the rollout workers talk
to the task server on :9999 and the vLLM replicas on their ports.

## Known limits
* Multi-node Ray is not wired (single node, 1..8 GPUs).
* `expandable_segments` allocator is incompatible with vLLM sleep mode; do not set it.
