# syntax=docker/dockerfile:1.7
# MAReasoning Flow-GRPO training image (AgentFlow fork + verl 0.5.0 + vLLM 0.9.2), 2026-09-30.
# One image for every role: trainer (verl/Ray, 1..N GPUs), rollout workers, vLLM helper/judge servers, eval harness.
# Build: docker/build.sh            Run: docker/entrypoint.sh {train|serve|eval|bash}   (see docker/README.md)
# Design constraints:
#   * CUDA 12.8 (works on driver >= 550 on mds1 and on RCP nodes; CUDA 13 does not on driver 550).
#   * python 3.12 + the EXACT package set of the validated mds1 venv (docker/requirements.lock).
#   * three verl 0.5.0 fixes applied as a patch (docker/verl-0.5.0-mareasoning.patch), marked PATCHED (MAReasoning).
#   * runs as any uid (RCP/RunAI sets the pod uid): everything under /workspace and /opt/venv is world-writable, HOME is inside /workspace.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates patch psmisc procps lsof iproute2 tini build-essential \
      libnuma1 libibverbs1 librdmacm1 openssh-client tmux vim-tiny \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.9.6 /uv /uvx /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy UV_HTTP_TIMEOUT=600
RUN uv python install 3.12 && uv venv /opt/venv --python 3.12 --seed
ENV PATH=/opt/venv/bin:$PATH VIRTUAL_ENV=/opt/venv
# 1) exact python stack (torch cu128 index for the +cu128 builds; unsafe-best-match lets the pin pick the right index)
COPY docker/requirements.lock /tmp/requirements.lock
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv -r /tmp/requirements.lock \
      --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
# 2) verl fixes (prefix-caching switch, FSDP1->FSDP2 optimizer resume, MERGE_DTYPE)
COPY docker/verl-0.5.0-mareasoning.patch /tmp/verl.patch
RUN cd /opt/venv/lib/python3.12/site-packages && patch -p1 --forward < /tmp/verl.patch \
    && grep -l "PATCHED (MAReasoning)" verl/workers/rollout/vllm_rollout/vllm_async_server.py verl/utils/checkpoint/fsdp_checkpoint_manager.py verl/model_merger/base_model_merger.py | wc -l | grep -q 3
# 3) the code (fork with PATCHED (MAReasoning) edits), editable so a bind-mount of a newer checkout also works
WORKDIR /workspace/AgentFlow
COPY . /workspace/AgentFlow
RUN uv pip install --python /opt/venv --no-deps -e /workspace/AgentFlow -e /workspace/AgentFlow/agentflow
# 4) any-uid friendliness + defaults. Data, HF cache, checkpoints and run logs are volumes (see README).
ENV HOME=/workspace/home PYTHONPATH=/workspace/AgentFlow PYTHONUNBUFFERED=1 VLLM_USE_V1=1 \
    HF_HOME=/data/hf AF_RUNS=/data/runs TOKENIZERS_PARALLELISM=false
RUN mkdir -p /workspace/home /data/hf /data/runs /data/ckpt && chmod -R a+rwX /workspace /opt/venv /data \
    && chmod +x /workspace/AgentFlow/docker/entrypoint.sh
ARG GIT_SHA=unknown
LABEL org.opencontainers.image.title="agentflow-train" org.opencontainers.image.revision=$GIT_SHA \
      org.opencontainers.image.description="MAReasoning Flow-GRPO: AgentFlow fork + verl 0.5.0 + vLLM 0.9.2, CUDA 12.8"
ENV AF_IMAGE_GIT_SHA=$GIT_SHA
ENTRYPOINT ["/usr/bin/tini", "--", "/workspace/AgentFlow/docker/entrypoint.sh"]
CMD ["bash"]
