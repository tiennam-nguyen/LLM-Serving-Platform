"""Structured JSON logging utility for the LLM Serving Platform."""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict

logger = logging.getLogger("llm_serving_platform.gateway")


def redact_prompt_text(message: str, payload: dict[str, Any]) -> str:
    """Remove message contents echoed by upstream errors before storing telemetry."""
    contents: set[str] = set()
    for item in payload.get("messages", []):
        content = item.get("content") if isinstance(item, dict) else None
        if isinstance(content, str) and content:
            contents.update(
                (content, json.dumps(content)[1:-1], json.dumps(content, ensure_ascii=False)[1:-1])
            )
    for content in sorted(contents, key=len, reverse=True):
        message = message.replace(content, "[redacted]")
    return message


def log_structured_event(
    event: str,
    request_id: str,
    client_ip: str,
    model: str = "",
    status: str = "ok",
    worker_id: str = "",
    extra: Dict[str, Any] | None = None,
) -> None:
    """Emit a structured JSON log record for end-to-end request correlation."""
    log_data: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "request_id": request_id,
        "client_ip": client_ip,
        "model": model,
        "worker_id": worker_id,
        "status": status,
    }
    if extra:
        log_data.update(extra)
    logger.info(json.dumps(log_data, default=str))
