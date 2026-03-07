#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export DISHSPACE_ROOT="$ROOT_DIR"
export RUNPOD_VOLUME_ROOT="${RUNPOD_VOLUME_ROOT:-/runpod-volume/dishspace}"
export DATA_DIR="${DATA_DIR:-$RUNPOD_VOLUME_ROOT/data}"
export MODELS_DIR="${MODELS_DIR:-$RUNPOD_VOLUME_ROOT/models}"
export CACHE_DIR="${CACHE_DIR:-$RUNPOD_VOLUME_ROOT/cache}"
export HF_HOME="${HF_HOME:-$CACHE_DIR/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export DEVICE="${DEVICE:-cuda}"

mkdir -p "$DATA_DIR" "$MODELS_DIR" "$CACHE_DIR" "$HF_HOME" "$TRANSFORMERS_CACHE" "$HUGGINGFACE_HUB_CACHE"

exec python -m uvicorn src.api.server:app --host 0.0.0.0 --port "${API_PORT:-8000}"