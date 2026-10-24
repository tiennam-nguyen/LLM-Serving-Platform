"""SSE stream orchestrator with Week 3 trace milestone instrumentation."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from llm_serving_platform.common.logging import log_structured_event
from llm_serving_platform.telemetry.trace import InMemoryTraceStore, RequestTrace
from llm_serving_platform.workers.base import InferenceWorker, WorkerException


def _inspect_sse_chunk(sse_chunk: str) -> tuple[bool, int | None]:
    """Inspect an SSE chunk for non-empty assistant content and explicit completion_tokens usage."""
    has_content = False
    completion_tokens: int | None = None

    for raw_line in sse_chunk.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        data_str = line.removeprefix("data:").strip()
        if not data_str or data_str == "[DONE]":
            continue
        try:
            parsed = json.loads(data_str)
            if not isinstance(parsed, dict) or "error" in parsed:
                continue

            choices = parsed.get("choices")
            if isinstance(choices, list) and choices:
                first_choice = choices[0]
                if isinstance(first_choice, dict):
                    delta = first_choice.get("delta", {})
                    if isinstance(delta, dict) and delta.get("content"):
                        has_content = True
            elif "token" in parsed and parsed.get("token"):
                has_content = True

            usage = parsed.get("usage")
            if isinstance(usage, dict):
                comp = usage.get("completion_tokens")
                if isinstance(comp, int) and not isinstance(comp, bool) and comp >= 0:
                    completion_tokens = comp
        except json.JSONDecodeError:
            continue

    return has_content, completion_tokens


def _is_content_chunk(sse_chunk: str) -> bool:
    """Return True if an SSE chunk carries non-empty assistant delta content or mock token."""
    has_content, _ = _inspect_sse_chunk(sse_chunk)
    return has_content


async def safe_event_generator(
    worker: InferenceWorker,
    payload: dict[str, Any],
    request_id: str,
    client_ip: str,
    trace: RequestTrace | None = None,
    trace_store: InMemoryTraceStore | None = None,
) -> AsyncIterator[str]:
    """Stream SSE events from worker, recording Week 3 trace milestones and handling errors."""
    model_name = payload.get("model", "unknown")
    worker_id = getattr(worker, "worker_id", "worker-vllm-0")

    if trace is None:
        trace = RequestTrace(request_id=request_id, model=model_name, worker_id=worker_id)
        trace.mark_arrival()
        trace.mark_routing_start()
        trace.mark_routing_end(worker_id=worker_id)

    trace.mark_worker_start()
    if trace_store is not None:
        trace_store.save(trace)

    try:
        async for chunk in worker.stream_chat(payload, request_id):
            has_content, completion_tokens = _inspect_sse_chunk(chunk)
            if has_content:
                trace.mark_first_token()
                trace.token_chunks += 1
            if completion_tokens is not None:
                trace.completion_tokens = completion_tokens
            yield chunk

        trace.mark_last_token(status="completed")
        if trace_store is not None:
            trace_store.save(trace)

        log_structured_event(
            event="chat_stream_completed",
            request_id=request_id,
            client_ip=client_ip,
            model=model_name,
            worker_id=worker_id,
            status="completed",
            extra=trace.to_dict(),
        )
    except asyncio.CancelledError:
        trace.mark_last_token(status="cancelled", error="Client disconnected")
        if trace_store is not None:
            trace_store.save(trace)

        log_structured_event(
            event="chat_stream_cancelled",
            request_id=request_id,
            client_ip=client_ip,
            model=model_name,
            worker_id=worker_id,
            status="cancelled",
            extra=trace.to_dict(),
        )
        raise
    except WorkerException as e:
        trace.mark_last_token(status="error", error=str(e))
        if trace_store is not None:
            trace_store.save(trace)

        log_structured_event(
            event="chat_stream_worker_error",
            request_id=request_id,
            client_ip=client_ip,
            model=model_name,
            worker_id=worker_id,
            status="error",
            extra={"error": str(e), "code": e.status_code, **trace.to_dict()},
        )
        err_json = json.dumps({"error": {"message": str(e), "code": e.status_code}})
        yield f"data: {err_json}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        trace.mark_last_token(status="error", error=str(e))
        if trace_store is not None:
            trace_store.save(trace)

        log_structured_event(
            event="chat_stream_internal_error",
            request_id=request_id,
            client_ip=client_ip,
            model=model_name,
            worker_id=worker_id,
            status="error",
            extra={"error": str(e), "code": 500, **trace.to_dict()},
        )
        err_json = json.dumps({"error": {"message": str(e), "code": 500}})
        yield f"data: {err_json}\n\n"
        yield "data: [DONE]\n\n"
