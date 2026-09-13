"""Mock implementations for workers and external dependencies."""

import asyncio
import json
from typing import Any, AsyncIterator, Dict, List, Optional

from llm_serving_platform.workers.base import (
    InferenceWorker,
    WorkerException,
)


class MockInferenceWorker(InferenceWorker):
    """In-memory mock inference worker for fast, deterministic unit & integration tests."""

    def __init__(
        self,
        canned_chunks: Optional[List[str]] = None,
        raise_error_before_stream: Optional[WorkerException] = None,
        raise_error_after_tokens: Optional[int] = None,
        mid_stream_error: Optional[WorkerException] = None,
        chunk_delay_s: float = 0.0,
        is_healthy: bool = True,
    ) -> None:
        self.canned_chunks = canned_chunks or ["Hello", " world", " from", " mock", "!"]
        self.raise_error_before_stream = raise_error_before_stream
        self.raise_error_after_tokens = raise_error_after_tokens
        self.mid_stream_error = mid_stream_error
        self.chunk_delay_s = chunk_delay_s
        self.is_healthy = is_healthy

        # Tracking for assertions
        self.received_requests: List[Dict[str, Any]] = []
        self.received_request_ids: List[str] = []

    async def stream_chat(
        self, payload: Dict[str, Any], request_id: str
    ) -> AsyncIterator[str]:
        self.received_requests.append(payload)
        self.received_request_ids.append(request_id)

        if self.raise_error_before_stream:
            raise self.raise_error_before_stream

        model = payload.get("model", "mock-model")

        for idx, chunk_text in enumerate(self.canned_chunks):
            if self.chunk_delay_s > 0:
                await asyncio.sleep(self.chunk_delay_s)

            if (
                self.raise_error_after_tokens is not None
                and idx >= self.raise_error_after_tokens
                and self.mid_stream_error is not None
            ):
                raise self.mid_stream_error

            sse_data = {
                "id": f"chatcmpl-mock-{idx}",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk_text},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(sse_data)}\n\n"

        yield "data: [DONE]\n\n"

    async def check_health(self) -> bool:
        return self.is_healthy
