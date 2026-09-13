#!/usr/bin/env bash
set -e
GATEWAY_URL="${1:-http://127.0.1.1:8080}"

echo "[*] Kiểm tra Gateway Health tại ${GATEWAY_URL}/health..."
if ! curl -s -f "${GATEWAY_URL}/health" > /dev/null; then
    echo "[-] LỖI: Không kết nối được Gateway hoặc worker báo 503."
    exit 1
fi

echo "[+] Gateway OK. Đang chạy 30 requests nghiệm thu..."
exec .venv/bin/python infra/scripts/verify_week2_exit_gate.py --gateway-url "${GATEWAY_URL}"
