"""Structured JSON logging utility for the LLM Serving Platform."""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict

logger = logging.getLogger("llm_serving_platform.gateway")


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
