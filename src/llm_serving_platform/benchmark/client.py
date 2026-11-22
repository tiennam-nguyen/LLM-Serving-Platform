"""HTTP-only client for interacting with the Gateway public data, control, and metrics APIs."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

TERMINAL_TRACE_STATUSES = frozenset({"completed", "error", "cancelled"})


class RunnerFatalError(RuntimeError):
    """Controlled infrastructure or contract error that aborts an experiment run."""

    def __init__(self, failure_type: str, message: str) -> None:
        super().__init__(message)
        self.failure_type = failure_type
        self.message = message


@dataclass(frozen=True)
class ChatCallResult:
    """Observed HTTP and SSE outcome of one POST /v1/chat/completions call."""

    status_code: int
    request_id: str
    response_request_id: str | None
    response_worker_id: str | None
    content_chunks: int
    saw_done: bool
    saw_sse_error: bool
    sse_error_code: int | None


def _parse_sse_data_line(line: str) -> tuple[bool, bool, bool, int | None]:
    """Return (is_done, has_content, has_error, error_code) for a stripped SSE line."""
    if not line.startswith("data:"):
        return False, False, False, None
    data_str = line.removeprefix("data:").strip()
    if not data_str:
        return False, False, False, None
    if data_str == "[DONE]":
        return True, False, False, None
    try:
        parsed = json.loads(data_str)
    except json.JSONDecodeError:
        return False, False, False, None
    if not isinstance(parsed, dict):
        return False, False, False, None
    if "error" in parsed:
        code: int | None = None
        err = parsed.get("error")
        if isinstance(err, dict):
            raw_code = err.get("code")
            if isinstance(raw_code, int) and not isinstance(raw_code, bool):
                code = raw_code
        return False, False, True, code

    has_content = False
    choices = parsed.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            delta = first.get("delta")
            if isinstance(delta, dict) and delta.get("content"):
                has_content = True
    elif parsed.get("token"):
        has_content = True
    return False, has_content, False, None


class GatewayHTTPClient:
    """Interact with the Gateway strictly through its public HTTP endpoints."""

    def __init__(
        self,
        gateway_url: str,
        *,
        timeout_s: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._gateway_url = gateway_url.rstrip("/")
        self._timeout_s = timeout_s
        self._external_client = client
        self._owned_client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> GatewayHTTPClient:
        if self._external_client is None:
            timeout = httpx.Timeout(self._timeout_s, connect=10.0)
            self._owned_client = httpx.AsyncClient(
                base_url=self._gateway_url,
                timeout=timeout,
                trust_env=False,
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._owned_client is not None:
            await self._owned_client.aclose()
            self._owned_client = None

    @property
    def _client(self) -> httpx.AsyncClient:
        client = self._external_client or self._owned_client
        if client is None:
            raise RuntimeError("GatewayHTTPClient must be used inside an async context manager")
        return client

    async def check_health(self) -> dict[str, Any]:
        """Verify Gateway reachability and healthy status via GET /health."""
        try:
            response = await self._client.get("/health")
        except (httpx.RequestError, httpx.TimeoutException) as exc:
            raise RunnerFatalError("gateway_unreachable", "Gateway /health unreachable") from exc

        if response.status_code != 200:
            raise RunnerFatalError(
                "gateway_unhealthy",
                f"Gateway /health returned HTTP {response.status_code}",
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise RunnerFatalError("contract_error", "Gateway /health returned non-JSON") from exc

        if not isinstance(body, dict) or body.get("status") != "healthy":
            raise RunnerFatalError("gateway_unhealthy", "Gateway /health reported unhealthy state")
        return body

    async def list_workers(self) -> list[dict[str, Any]]:
        """Inspect worker topology via GET /control/v1/workers."""
        try:
            response = await self._client.get("/control/v1/workers")
        except (httpx.RequestError, httpx.TimeoutException) as exc:
            raise RunnerFatalError(
                "gateway_unreachable", "Gateway /control/v1/workers unreachable"
            ) from exc

        if response.status_code != 200:
            raise RunnerFatalError(
                "contract_error",
                f"Gateway /control/v1/workers returned HTTP {response.status_code}",
            )
        try:
            workers = response.json()
        except ValueError as exc:
            raise RunnerFatalError(
                "contract_error", "Gateway /control/v1/workers returned non-JSON"
            ) from exc

        if not isinstance(workers, list) or not all(isinstance(w, dict) for w in workers):
            raise RunnerFatalError(
                "contract_error", "Gateway /control/v1/workers did not return a worker list"
            )
        return workers

    async def stream_chat_completion(
        self,
        payload: dict[str, Any],
        request_id: str,
    ) -> ChatCallResult:
        """Send POST /v1/chat/completions with stream=True and consume the full SSE body."""
        headers = {
            "X-Request-ID": request_id,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        try:
            async with self._client.stream(
                "POST",
                "/v1/chat/completions",
                json=payload,
                headers=headers,
            ) as response:
                status_code = response.status_code
                resp_req_id = response.headers.get("X-Request-ID")
                resp_worker_id = response.headers.get("X-Worker-ID")
                content_chunks = 0
                saw_done = False
                saw_sse_error = False
                sse_error_code: int | None = None

                if status_code == 200:
                    async for raw_line in response.aiter_lines():
                        line = raw_line.strip()
                        if not line:
                            continue
                        is_done, has_content, has_error, err_code = _parse_sse_data_line(line)
                        if is_done:
                            saw_done = True
                        if has_content:
                            content_chunks += 1
                        if has_error:
                            saw_sse_error = True
                            if err_code is not None:
                                sse_error_code = err_code
                else:
                    await response.aread()
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError) as exc:
            raise RunnerFatalError(
                "gateway_unreachable", "Gateway unreachable during chat"
            ) from exc
        except httpx.TimeoutException as exc:
            raise RunnerFatalError("gateway_timeout", "Gateway request timed out") from exc
        except httpx.RequestError as exc:
            raise RunnerFatalError(
                "gateway_transport_error", "Gateway transport error during chat"
            ) from exc

        if resp_req_id is not None and resp_req_id != request_id:
            raise RunnerFatalError(
                "contract_error",
                f"Gateway returned mismatched X-Request-ID {resp_req_id!r} != {request_id!r}",
            )

        return ChatCallResult(
            status_code=status_code,
            request_id=request_id,
            response_request_id=resp_req_id,
            response_worker_id=resp_worker_id,
            content_chunks=content_chunks,
            saw_done=saw_done,
            saw_sse_error=saw_sse_error,
            sse_error_code=sse_error_code,
        )

    async def fetch_terminal_trace(
        self,
        request_id: str,
        *,
        timeout_s: float = 5.0,
        poll_interval_s: float = 0.02,
    ) -> dict[str, Any]:
        """Poll GET /v1/traces/{request_id} with a bounded deadline until terminal."""
        encoded = quote(request_id, safe="")
        path = f"/v1/traces/{encoded}"
        deadline = asyncio.get_running_loop().time() + timeout_s

        while True:
            try:
                response = await self._client.get(path)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError) as exc:
                raise RunnerFatalError(
                    "gateway_unreachable", "Gateway unreachable while fetching trace"
                ) from exc
            except (httpx.TimeoutException, httpx.RequestError) as exc:
                raise RunnerFatalError(
                    "gateway_transport_error", "Gateway transport error while fetching trace"
                ) from exc

            if response.status_code == 200:
                try:
                    trace = response.json()
                except ValueError as exc:
                    raise RunnerFatalError(
                        "contract_error", "Gateway trace endpoint returned non-JSON"
                    ) from exc
                if not isinstance(trace, dict) or trace.get("request_id") != request_id:
                    raise RunnerFatalError(
                        "contract_error", "Gateway trace payload failed request_id correlation"
                    )
                if trace.get("status") in TERMINAL_TRACE_STATUSES:
                    return trace
            elif response.status_code != 404:
                raise RunnerFatalError(
                    "contract_error",
                    f"Gateway trace endpoint returned unexpected HTTP {response.status_code}",
                )

            if asyncio.get_running_loop().time() >= deadline:
                raise RunnerFatalError(
                    "trace_timeout",
                    f"Timed out waiting for terminal trace for {request_id}",
                )
            await asyncio.sleep(poll_interval_s)

    async def scrape_metrics(self) -> str:
        """Scrape raw Prometheus exposition text from GET /metrics."""
        try:
            response = await self._client.get("/metrics")
        except (httpx.RequestError, httpx.TimeoutException) as exc:
            raise RunnerFatalError("gateway_unreachable", "Gateway /metrics unreachable") from exc

        if response.status_code != 200:
            raise RunnerFatalError(
                "contract_error", f"Gateway /metrics returned HTTP {response.status_code}"
            )
        return response.text
