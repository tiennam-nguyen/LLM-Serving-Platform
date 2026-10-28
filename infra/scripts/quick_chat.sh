#!/usr/bin/env bash
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

GATEWAY_URL="${1:-${GATEWAY_URL:-http://127.0.0.1:8080}}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
PROMPT="${2:-Giải thích ngắn gọn cơ chế KV-cache trong 2 câu.}"
REQ_ID="cli-test-$(date +%s)"

echo "[*] Gửi request: \"${PROMPT}\" (ID: ${REQ_ID})"
curl -N -s -X POST "${GATEWAY_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: ${REQ_ID}" \
  -d "{
    \"model\": \"${MODEL_NAME}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"${PROMPT}\"}],
    \"stream\": true,
    \"max_tokens\": 100
  }"
echo ""
