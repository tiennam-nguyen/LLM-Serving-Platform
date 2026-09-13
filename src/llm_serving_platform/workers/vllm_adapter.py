"""vLLM engine worker adapter implementation."""

from typing import Any, AsyncIterator, Dict, Optional

import httpx

from llm_serving_platform.workers.base import (
    InferenceWorker,
    WorkerException,
    WorkerTimeoutError,
    WorkerUnavailableError,
)


class VLLMWorkerAdapter(InferenceWorker):
    """Adapter facilitating communication with a vLLM OpenAI-compatible API server."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        client: Optional[httpx.AsyncClient] = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client
        self.timeout = timeout

    @property
    def client(self) -> httpx.AsyncClient:
        """Return the injected HTTP client or raise an error if not configured."""
        if self._client is None:
            raise RuntimeError("HTTP client is not initialized on VLLMWorkerAdapter.")
        return self._client

    @client.setter
    def client(self, value: httpx.AsyncClient) -> None:
        self._client = value

    async def stream_chat(
        self, payload: Dict[str, Any], request_id: str
    ) -> AsyncIterator[str]:
        """Forward chat completion requests to vLLM and stream SSE chunks.

        Uses async context manager stream to ensure safe connection cleanup.
        """
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "X-Request-ID": request_id,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        try:
            async with self.client.stream(
                "POST",
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            ) as response:
                if response.status_code != 200:
                    error_bytes = await response.aread()
                    error_msg = error_bytes.decode("utf-8", errors="replace")
                    raise WorkerUnavailableError(
                        f"vLLM worker returned HTTP {response.status_code}: {error_msg}",
                        status_code=response.status_code,
                    )

                async for raw_line in response.aiter_lines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    if line.startswith("data:"):
                        yield f"{line}\n\n"
                    else:
                        yield f"data: {line}\n\n"

        except httpx.TimeoutException as exc:
            raise WorkerTimeoutError(
                f"vLLM request timed out after {self.timeout}s: {exc}"
            ) from exc
        except WorkerException:
            raise
        except httpx.RequestError as exc:
            raise WorkerUnavailableError(
                f"Failed to communicate with vLLM worker: {exc}"
            ) from exc

    async def check_health(self) -> bool:
        """Check if the vLLM backend server is healthy."""
        url = f"{self.base_url}/health"
        try:
            response = await self.client.get(url, timeout=min(5.0, self.timeout))
            return response.status_code == 200
        except (httpx.RequestError, httpx.TimeoutException):
            return False
