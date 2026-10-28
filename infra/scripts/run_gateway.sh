#!/usr/bin/env bash
set -e
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

for arg in "$@"; do
    if [ "$arg" = "--no-gpu" ] || [ "$arg" = "--mock" ]; then
        export NO_GPU=1
    fi
done

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000}"
export GATEWAY_PORT="${GATEWAY_PORT:-8080}"
export GATEWAY_HOST="${GATEWAY_HOST:-0.0.0.0}"

echo "[*] Khởi động Gateway tại ${GATEWAY_HOST}:${GATEWAY_PORT}"
if [ "${NO_GPU:-0}" = "1" ]; then
    echo "[*] Chế độ Worker: NO-GPU (MockInferenceWorker in-memory)"
else
    echo "[*] Chế độ Worker: LIVE-GPU (vLLM tại ${VLLM_BASE_URL})"
fi

if [ -x ".venv/bin/uvicorn" ]; then
    exec .venv/bin/uvicorn llm_serving_platform.gateway.main:app --host "${GATEWAY_HOST}" --port "${GATEWAY_PORT}"
else
    exec python3 -m uvicorn llm_serving_platform.gateway.main:app --host "${GATEWAY_HOST}" --port "${GATEWAY_PORT}"
fi
