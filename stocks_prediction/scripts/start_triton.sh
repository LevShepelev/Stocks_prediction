#!/usr/bin/env bash
# scripts/start_triton.sh
set -euo pipefail

# Absolute path to repo root
ROOT_DIR="$(git rev-parse --show-toplevel)"

# Default ports can be overridden via env vars
HTTP_PORT="${HTTP_PORT:-8011}"
GRPC_PORT="${GRPC_PORT:-8012}"
METRICS_PORT="${METRICS_PORT:-8013}"

# Launch (remove --rm if you want persistent logs)
docker run --rm --gpus=all --name triton_server \
  -p"${HTTP_PORT}":8000 -p"${GRPC_PORT}":8001 -p"${METRICS_PORT}":8002 \
  -v "${ROOT_DIR}/model_repo:/models" \
  nvcr.io/nvidia/tritonserver:25.04-py3 \
  tritonserver --model-repository=/models "$@"
