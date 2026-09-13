"""FastAPI API Gateway entrypoint for LLM Serving Platform."""

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from llm_serving_platform.gateway.schemas import ChatCompletionRequest
from llm_serving_platform.workers.base import InferenceWorker, WorkerException
from llm_serving_platform.workers.vllm_adapter import VLLMWorkerAdapter

logger = logging.getLogger("llm_serving_platform.gateway")


def log_structured_event(
    event: str,
    request_id: str,
    client_ip: str,
    model: str = "",
    status: str = "ok",
    extra: Dict[str, Any] | None = None,
) -> None:
    """Emit a structured JSON log record."""
    log_data: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "request_id": request_id,
        "client_ip": client_ip,
        "model": model,
        "status": status,
    }
    if extra:
        log_data.update(extra)
    logger.info(json.dumps(log_data, default=str))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the lifecycle of the shared httpx client and default worker."""
    limits = httpx.Limits(max_keepalive_connections=50, max_connections=200)
    timeout = httpx.Timeout(60.0, connect=5.0)
    client = httpx.AsyncClient(limits=limits, timeout=timeout)
    app.state.client = client

    base_url = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000")
    app.state.worker = VLLMWorkerAdapter(base_url=base_url, client=client)

    yield

    await client.aclose()


def get_worker(request: Request) -> InferenceWorker:
    """Dependency provider for the inference worker."""
    worker: InferenceWorker = getattr(request.app.state, "worker", None)
    if worker is None:
        raise RuntimeError("Inference worker is not initialized on app state.")
    return worker


async def safe_event_generator(
    worker: InferenceWorker, payload: Dict[str, Any], request_id: str, client_ip: str
) -> AsyncIterator[str]:
    """Safely stream SSE events from worker, catching errors and client aborts."""
    model_name = payload.get("model", "unknown")
    try:
        async for chunk in worker.stream_chat(payload, request_id):
            yield chunk
        log_structured_event(
            event="chat_stream_completed",
            request_id=request_id,
            client_ip=client_ip,
            model=model_name,
            status="completed",
        )
    except asyncio.CancelledError:
        # Client closed tab/disconnected, clean up and exit immediately
        log_structured_event(
            event="chat_stream_cancelled",
            request_id=request_id,
            client_ip=client_ip,
            model=model_name,
            status="cancelled",
        )
        raise
    except WorkerException as e:
        log_structured_event(
            event="chat_stream_worker_error",
            request_id=request_id,
            client_ip=client_ip,
            model=model_name,
            status="error",
            extra={"error": str(e), "code": e.status_code},
        )
        err_json = json.dumps({"error": {"message": str(e), "code": e.status_code}})
        yield f"data: {err_json}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        log_structured_event(
            event="chat_stream_internal_error",
            request_id=request_id,
            client_ip=client_ip,
            model=model_name,
            status="error",
            extra={"error": str(e), "code": 500},
        )
        err_json = json.dumps({"error": {"message": str(e), "code": 500}})
        yield f"data: {err_json}\n\n"
        yield "data: [DONE]\n\n"


def create_app() -> FastAPI:
    """Application factory for the API Gateway."""
    app = FastAPI(
        title="LLM Serving Platform - API Gateway",
        description="Data Plane API Gateway for observable multi-GPU LLM inference.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        request_id = getattr(request.state, "request_id", "unknown")
        client_ip = request.client.host if request.client else "unknown"
        serializable_details = [
            {
                "type": err.get("type"),
                "loc": err.get("loc"),
                "msg": err.get("msg"),
            }
            for err in exc.errors()
        ]
        log_structured_event(
            event="validation_error",
            request_id=request_id,
            client_ip=client_ip,
            status="error",
            extra={"details": serializable_details},
        )
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "Invalid request payload",
                    "details": serializable_details,
                }
            },
        )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        req_id = request.headers.get("X-Request-ID")
        if not req_id or not req_id.strip():
            req_id = str(uuid.uuid4())
        request.state.request_id = req_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response

    @app.get("/health")
    async def health(worker: InferenceWorker = Depends(get_worker)):
        """Check the health status of the Gateway and backend worker."""
        is_healthy = await worker.check_health()
        if not is_healthy:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "worker": "down"},
            )
        return {"status": "healthy", "worker": "up"}

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request,
        request_payload: ChatCompletionRequest,
        worker: InferenceWorker = Depends(get_worker),
    ):
        """OpenAI-compatible streaming chat completion endpoint."""
        if not request_payload.stream:
            raise HTTPException(
                status_code=400,
                detail="Only streaming mode (stream=True) is currently supported.",
            )

        request_id = request.state.request_id
        client_ip = request.client.host if request.client else "unknown"
        model_name = request_payload.model

        # Serialize payload according to Critical Invariant 4
        payload = request_payload.model_dump(exclude_none=True)

        log_structured_event(
            event="chat_stream_started",
            request_id=request_id,
            client_ip=client_ip,
            model=model_name,
            status="started",
        )

        return StreamingResponse(
            safe_event_generator(worker, payload, request_id, client_ip),
            media_type="text/event-stream",
            headers={"X-Request-ID": request_id},
        )

    return app


app = create_app()
