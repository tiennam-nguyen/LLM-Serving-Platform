#!/usr/bin/env bash
GATEWAY_URL="${1:-http://127.0.1.1:8080}"
PROMPT="${2:-Giải thích ngắn gọn cơ chế KV-cache trong 2 câu.}"
REQ_ID="cli-test-$(date +%s)"

echo "[*] Gửi request: \"${PROMPT}\" (ID: ${REQ_ID})"
curl -N -s -X POST "${GATEWAY_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: ${REQ_ID}" \
  -d "{
    \"model\": \"Qwen/Qwen2.5-0.5B-Instruct\",
    \"messages\": [{\"role\": \"user\", \"content\": \"${PROMPT}\"}],
    \"stream\": true,
    \"max_tokens\": 100
  }"
echo ""
