#!/bin/bash
# MAReasoning Exp 5 (2026-10-02): build the UPSTREAM AgentFlow environment exactly as its setup.sh + scripts/setup_stable_gpu.sh
# prescribe (uv venv python 3.11; torch 2.7.0 cu128, transformers 4.53.3, flash-attn 2.8.1, vllm 0.9.2, verl 0.5.0).
# Only deviation: flash-attn from the matching prebuilt wheel instead of a source build (same version; a source build takes hours).
set -eu
export PATH=$HOME/.local/bin:$PATH UV_HTTP_TIMEOUT=600 HF_HOME=/data/sakhaei/hf
cd /data/sakhaei/code/AgentFlow-upstream
log(){ echo "$(date '+%F %T') $*"; }
log "uv venv python 3.11"
uv venv -p 3.11 .venv
source .venv/bin/activate
log "agentflow/requirements.txt"
(cd agentflow && uv pip install -r requirements.txt && uv pip install --no-deps -e .)
log "project -e . + setup.sh extras"
uv pip install -e .
uv pip install dashscope fire
uv pip install "autogen-agentchat" "autogen-ext[openai]"
uv pip install "litellm[proxy]"
uv pip install mcp openai-agents
uv pip install langgraph "langchain[openai]" langchain-community langchain-text-splitters
uv pip install sqlparse nltk yq
log "scripts/setup_stable_gpu.sh pins"
uv pip install --no-cache-dir packaging ninja numpy pandas ipython ipykernel gdown wheel setuptools
uv pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --no-cache-dir transformers==4.53.3
uv pip install --no-cache-dir "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
uv pip install --no-cache-dir vllm==0.9.2
uv pip install --no-cache-dir verl==0.5.0
uv pip install --no-cache-dir -e ".[dev,agent]"
log "versions:"
python -c "import torch,vllm,verl,transformers,flash_attn,agentops; print('torch',torch.__version__,'vllm',vllm.__version__,'verl',verl.__version__,'transformers',transformers.__version__,'flash_attn',flash_attn.__version__,'agentops',agentops.__version__)"
python -c "import agentflow, agentflow.verl; print('agentflow import ok')" || echo "agentflow.verl import FAILED"
log "######## UPSTREAM ENV BUILD DONE ########"
