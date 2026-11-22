"""Structured JSON logging utility and controlled telemetry error classification."""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict

from llm_serving_platform.workers.base import (
    WorkerException,
    WorkerTimeoutError,
    WorkerUnavailableError,
)

logger = logging.getLogger("llm_serving_platform.gateway")


@dataclass(frozen=True)
class SafeTelemetryError:
    """Privacy-safe error classification persisted in traces and structured logs."""

    error: str
    error_type: str
    error_code: int | None = None


def classify_telemetry_error(
    exc: BaseException,
    *,
    default_error: str = "internal_error",
    default_code: int | None = 500,
) -> SafeTelemetryError:
    """Map an exception into a controlled vocabulary without persisting raw upstream text."""
    if isinstance(exc, (asyncio.CancelledError, GeneratorExit)):
        return SafeTelemetryError(
            error="client_disconnected",
            error_type="ClientDisconnected",
            error_code=None,
        )
    if isinstance(exc, WorkerTimeoutError):
        return SafeTelemetryError(
            error="worker_timeout",
            error_type=type(exc).__name__,
            error_code=exc.status_code,
        )
    if isinstance(exc, WorkerUnavailableError):
        return SafeTelemetryError(
            error="worker_unavailable",
            error_type=type(exc).__name__,
            error_code=exc.status_code,
        )
    if isinstance(exc, WorkerException):
        return SafeTelemetryError(
            error="worker_error",
            error_type=type(exc).__name__,
            error_code=exc.status_code,
        )
    return SafeTelemetryError(
        error=default_error,
        error_type=type(exc).__name__,
        error_code=default_code,
    )


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
