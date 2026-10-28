#!/usr/bin/env bash
set -e
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"

uv run python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --dtype float16 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 2048 \
    --swap-space 1 \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}"
