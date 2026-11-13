#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if (( $# > 1 )); then echo "Usage: $0 [trusted-env-file]" >&2; exit 2; fi
ENV_FILE="${1:-$ROOT/.env.week5}"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    source "$ENV_FILE"
    set +a
elif (( $# )); then
    echo "Env file not found: $ENV_FILE" >&2; exit 2
fi
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-${WORKER_PORT:-8000}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
SWAP_SPACE_GB="${SWAP_SPACE_GB:-1}"
printf 'Model: %s\nBind: %s:%s\nCUDA_VISIBLE_DEVICES: %s\n' \
    "$MODEL_NAME" "$VLLM_HOST" "$VLLM_PORT" "$CUDA_VISIBLE_DEVICES"
printf 'dtype: float16 | memory: %s | context: %s | swap GiB: %s | TP: 1\n' \
    "$GPU_MEMORY_UTILIZATION" "$MAX_MODEL_LEN" "$SWAP_SPACE_GB"
exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" --dtype float16 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" --swap-space "$SWAP_SPACE_GB" \
    --host "$VLLM_HOST" --port "$VLLM_PORT" --tensor-parallel-size 1
