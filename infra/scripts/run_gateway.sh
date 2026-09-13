#!/usr/bin/env bash
set -e
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.1.1:8000}"
export GATEWAY_PORT="${GATEWAY_PORT:-8080}"
export GATEWAY_HOST="${GATEWAY_HOST:-0.0.0.0}"

echo "[*] Khởi động Gateway tại ${GATEWAY_HOST}:${GATEWAY_PORT}"
echo "[*] Trỏ backend vLLM: ${VLLM_BASE_URL}"
exec .venv/bin/uvicorn llm_serving_platform.gateway.main:app --host "${GATEWAY_HOST}" --port "${GATEWAY_PORT}"
