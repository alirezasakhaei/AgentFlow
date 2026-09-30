#!/bin/bash
# Build the MAReasoning training image from the repo root. Tags: agentflow-train:<git sha> and :latest.
# Push (after `docker login registry.rcp.epfl.ch`): docker/build.sh push  -> registry.rcp.epfl.ch/criteria-feedback/agentflow-train:<sha>
set -eu
cd "$(dirname "$0")/.."
SHA=$(git rev-parse --short HEAD 2>/dev/null || echo nogit); IMG=agentflow-train; REG=${REG:-registry.rcp.epfl.ch/criteria-feedback}
DOCKER_BUILDKIT=1 docker build --platform linux/amd64 -f docker/Dockerfile --build-arg GIT_SHA=$SHA -t $IMG:$SHA -t $IMG:latest .
echo "built $IMG:$SHA"
if [ "${1:-}" = push ]; then docker tag $IMG:$SHA $REG/$IMG:$SHA; docker tag $IMG:$SHA $REG/$IMG:latest; docker push $REG/$IMG:$SHA; docker push $REG/$IMG:latest; fi
