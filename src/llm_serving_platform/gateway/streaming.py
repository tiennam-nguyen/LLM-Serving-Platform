"""SSE stream orchestrator with Week 3 trace milestone instrumentation."""

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

from anyio import CancelScope

from llm_serving_platform.common.logging import log_structured_event, redact_prompt_text
from llm_serving_platform.telemetry.trace import InMemoryTraceStore, RequestTrace
from llm_serving_platform.workers.base import InferenceWorker, WorkerException


def _inspect_sse_chunk(sse_chunk: str) -> tuple[bool, int | None, bool]:
    """Inspect an SSE chunk for non-empty assistant content and explicit completion_tokens usage."""
    has_content = False
    completion_tokens: int | None = None
    has_error = False

    for raw_line in sse_chunk.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        data_str = line.removeprefix("data:").strip()
        if not data_str or data_str == "[DONE]":
            continue
        try:
            parsed = json.loads(data_str)
            if not isinstance(parsed, dict):
                continue
            if "error" in parsed:
                has_error = True
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

    return has_content, completion_tokens, has_error


def _is_content_chunk(sse_chunk: str) -> bool:
    """Return True if an SSE chunk carries non-empty assistant delta content or mock token."""
    has_content, _, _ = _inspect_sse_chunk(sse_chunk)
    return has_content


async def safe_event_generator(
    worker: InferenceWorker,
    payload: dict[str, Any],
    request_id: str,
    client_ip: str,
    trace: RequestTrace | None = None,
    trace_store: InMemoryTraceStore | None = None,
    *,
    log_terminal: bool = True,
) -> AsyncGenerator[str, None]:
    """Stream SSE events from worker, recording Week 3 trace milestones and handling errors."""
    model_name = payload.get("model", "unknown")
    worker_id = getattr(worker, "worker_id", "worker-vllm-0")

    def emit_terminal(event: str, code: int | None = None) -> None:
        if log_terminal:
            log_structured_event(
                event=event,
                request_id=request_id,
                client_ip=client_ip,
                model=model_name,
                worker_id=worker_id,
                status=trace.status,
                extra={**trace.to_dict(), **({"code": code} if code is not None else {})},
            )

    if trace is None:
        trace = RequestTrace(request_id=request_id, model=model_name, worker_id=worker_id)
        trace.mark_arrival()
        trace.mark_routing_start()
        trace.mark_routing_end(worker_id=worker_id)

    trace.mark_worker_start()
    if trace_store is not None:
        trace_store.save(trace)

    stream = None
    worker_sent_error = False
    try:
        stream = worker.stream_chat(payload, request_id)
        async for chunk in stream:
            has_content, completion_tokens, has_error = _inspect_sse_chunk(chunk)
            worker_sent_error = worker_sent_error or has_error
            if has_content:
                trace.mark_first_token()
                trace.token_chunks += 1
            if completion_tokens is not None:
                trace.completion_tokens = completion_tokens
            yield chunk

        trace.mark_last_token(
            status="error" if worker_sent_error else "completed",
            error="Worker returned an SSE error" if worker_sent_error else None,
        )
        if trace_store is not None:
            trace_store.save(trace)

        emit_terminal("chat_stream_worker_error" if worker_sent_error else "chat_stream_completed")
    except (asyncio.CancelledError, GeneratorExit):
        trace.mark_last_token(status="cancelled", error="Client disconnected")
        if trace_store is not None:
            trace_store.save(trace)

        emit_terminal("chat_stream_cancelled")
        raise
    except WorkerException as e:
        trace.mark_last_token(status="error", error=redact_prompt_text(str(e), payload))
        if trace_store is not None:
            trace_store.save(trace)

        emit_terminal("chat_stream_worker_error", code=e.status_code)
        err_json = json.dumps({"error": {"message": str(e), "code": e.status_code}})
        yield f"data: {err_json}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        trace.mark_last_token(status="error", error=redact_prompt_text(str(e), payload))
        if trace_store is not None:
            trace_store.save(trace)

        emit_terminal("chat_stream_internal_error", code=500)
        err_json = json.dumps({"error": {"message": str(e), "code": 500}})
        yield f"data: {err_json}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        close_stream = getattr(stream, "aclose", None)
        if close_stream is not None:
            with CancelScope(shield=True):
                await close_stream()
