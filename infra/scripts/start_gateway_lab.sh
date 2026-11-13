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
export GATEWAY_HOST="${GATEWAY_HOST:-0.0.0.0}"
export GATEWAY_PORT="${GATEWAY_PORT:-8080}"
export HEALTH_CHECK_INTERVAL_S="${HEALTH_CHECK_INTERVAL_S:-2}"
export HEALTH_CHECK_TIMEOUT_S="${HEALTH_CHECK_TIMEOUT_S:-1}"
export VLLM_TIMEOUT_S="${VLLM_TIMEOUT_S:-60}"
# Prevent ambient mock-mode settings or a root .env from silently changing this lab.
export NO_GPU=0 WORKER_MODE=live ROUTING_POLICY=round_robin
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -z "${WORKER_ENDPOINTS:-}" ]]; then
    : "${HOST_A_IP:?Set HOST_A_IP and HOST_B_IP, or WORKER_ENDPOINTS}"
    : "${HOST_B_IP:?Set HOST_A_IP and HOST_B_IP, or WORKER_ENDPOINTS}"
    export HOST_A_IP HOST_B_IP
    python - <<'PY'
import ipaddress
import os
import sys
try:
    addresses = [ipaddress.IPv4Address(os.environ[k]) for k in ("HOST_A_IP", "HOST_B_IP")]
    if addresses[0] == addresses[1] or any(a.is_unspecified or a.is_multicast for a in addresses):
        raise ValueError("use two distinct unicast host addresses")
except ValueError:
    sys.exit("Set HOST_A_IP/HOST_B_IP to distinct IPv4 LAN addresses; replace template placeholders.")
PY
    WORKER_ENDPOINTS="${WORKER_A_ID:-worker-vllm-a}=http://$HOST_A_IP:${WORKER_PORT:-8000},${WORKER_B_ID:-worker-vllm-b}=http://$HOST_B_IP:${WORKER_PORT:-8000}"
fi
export WORKER_ENDPOINTS
# Validate before logging: credentials, query strings and fragments are not lab endpoints.
python - <<'PY'
import math
import os
import re
import sys
from urllib.parse import urlsplit
try:
    specs = os.environ["WORKER_ENDPOINTS"].split(",")
    if len(specs) != 2:
        raise ValueError("Week 5 needs exactly two id=http(s)://host:port endpoints")
    ids, urls = set(), set()
    for spec in specs:
        worker, url = spec.split("=", 1)
        parts = urlsplit(url)
        if (not re.fullmatch(r"[A-Za-z0-9_-]+", worker)
                or "REPLACE" in url.upper() or parts.scheme not in {"http", "https"}
                or not parts.hostname or parts.username is not None or parts.password is not None
                or parts.query or parts.fragment or parts.path not in {"", "/"}
                or parts.port is None or not 1 <= parts.port <= 65535):
            raise ValueError("invalid worker endpoint (use id=http(s)://host:port; no credentials)")
        ids.add(worker)
        urls.add(url.rstrip("/"))
    if len(ids) != 2 or len(urls) != 2:
        raise ValueError("worker IDs and endpoints must be distinct")
    if not 1 <= int(os.environ["GATEWAY_PORT"]) <= 65535:
        raise ValueError("invalid Gateway port")
    for key in ("HEALTH_CHECK_INTERVAL_S", "HEALTH_CHECK_TIMEOUT_S", "VLLM_TIMEOUT_S"):
        value = float(os.environ[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
except (ValueError, KeyError):
    sys.exit("Invalid lab configuration: check IDs, endpoints, ports and positive timeouts; no placeholders or credentials.")
PY
IFS=',' read -r WORKER_A WORKER_B <<< "$WORKER_ENDPOINTS"
printf 'Gateway        : %s:%s\nPolicy         : %s\nWorker A       : %s\nWorker B       : %s\n' \
    "$GATEWAY_HOST" "$GATEWAY_PORT" "$ROUTING_POLICY" "$WORKER_A" "$WORKER_B"
printf 'Health interval: %s\nHealth timeout : %s\nWorker timeout : %s\nModel          : %s\n' \
    "$HEALTH_CHECK_INTERVAL_S" "$HEALTH_CHECK_TIMEOUT_S" "$VLLM_TIMEOUT_S" "$MODEL_NAME"
exec python -m uvicorn llm_serving_platform.gateway.main:app \
    --host "$GATEWAY_HOST" --port "$GATEWAY_PORT"
