"""FastAPI API Gateway entrypoint for LLM Serving Platform."""

import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from llm_serving_platform.common.config import GatewayConfig
from llm_serving_platform.common.logging import log_structured_event
from llm_serving_platform.gateway.schemas import ChatCompletionRequest
from llm_serving_platform.gateway.streaming import safe_event_generator
from llm_serving_platform.routing.router import RoutingPolicy, SingleWorkerRouter
from llm_serving_platform.telemetry.trace import InMemoryTraceStore, RequestTrace
from llm_serving_platform.workers.base import InferenceWorker
from llm_serving_platform.workers.vllm_adapter import VLLMWorkerAdapter


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the lifecycle of the shared httpx client, default worker, and trace store."""
    config = GatewayConfig()
    limits = httpx.Limits(max_keepalive_connections=50, max_connections=200)
    timeout = httpx.Timeout(config.worker_timeout_s, connect=5.0)
    client = httpx.AsyncClient(limits=limits, timeout=timeout)

    app.state.config = config
    app.state.client = client
    app.state.worker = VLLMWorkerAdapter(
        base_url=config.vllm_base_url,
        client=client,
        timeout=config.worker_timeout_s,
        worker_id=config.worker_id,
    )
    if not hasattr(app.state, "trace_store"):
        app.state.trace_store = InMemoryTraceStore(max_size=500)

    yield

    await client.aclose()


async def get_worker(request: Request) -> InferenceWorker:
    """Async dependency provider for the inference worker (avoids threadpool hop)."""
    worker: InferenceWorker | None = getattr(request.app.state, "worker", None)
    if worker is None:
        cfg: GatewayConfig = getattr(request.app.state, "config", None) or GatewayConfig()
        client: httpx.AsyncClient | None = getattr(request.app.state, "client", None)
        if client is None:
            limits = httpx.Limits(max_keepalive_connections=50, max_connections=200)
            timeout = httpx.Timeout(cfg.worker_timeout_s, connect=5.0)
            client = httpx.AsyncClient(limits=limits, timeout=timeout)
            request.app.state.client = client
        worker = VLLMWorkerAdapter(
            base_url=cfg.vllm_base_url,
            client=client,
            timeout=cfg.worker_timeout_s,
            worker_id=cfg.worker_id,
        )
        request.app.state.worker = worker
    return worker


async def get_router(
    request: Request,
    worker: InferenceWorker = Depends(get_worker),
) -> RoutingPolicy:
    """Dependency provider for the routing policy, decoupled from HTTP controller logic."""
    router: RoutingPolicy | None = getattr(request.app.state, "router", None)
    if isinstance(router, SingleWorkerRouter) and router.worker is worker:
        return router
    if router is not None and not isinstance(router, SingleWorkerRouter):
        return router
    router = SingleWorkerRouter(worker)
    request.app.state.router = router
    return router


async def get_trace_store(request: Request) -> InMemoryTraceStore:
    """Async dependency provider for the in-memory request trace store."""
    store: InMemoryTraceStore | None = getattr(request.app.state, "trace_store", None)
    if store is None:
        store = InMemoryTraceStore(max_size=500)
        request.app.state.trace_store = store
    return store


def create_app(config: GatewayConfig | None = None) -> FastAPI:
    """Application factory for the API Gateway."""
    cfg = config or GatewayConfig()
    app = FastAPI(
        title="LLM Serving Platform - API Gateway",
        description="Data Plane API Gateway for observable multi-GPU LLM inference.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.config = cfg
    app.state.trace_store = InMemoryTraceStore(max_size=500)

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

    # Add CORSMiddleware after request_id_middleware so CORS wraps outermost
    # and attaches Access-Control-Allow-Origin even on error responses.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Accept", "X-Request-ID"],
        expose_headers=["X-Request-ID", "X-Worker-ID"],
    )

    @app.get("/health")
    async def health(worker: InferenceWorker = Depends(get_worker)):
        """Check the health status of the Gateway and backend worker."""
        worker_id = getattr(worker, "worker_id", "worker-vllm-0")
        is_healthy = await worker.check_health()
        if not is_healthy:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "worker": "down"},
                headers={"X-Worker-ID": worker_id},
            )
        return JSONResponse(
            status_code=200,
            content={"status": "healthy", "worker": "up"},
            headers={"X-Worker-ID": worker_id},
        )

    @app.get("/v1/traces")
    async def list_traces(trace_store: InMemoryTraceStore = Depends(get_trace_store)):
        """Return recent request traces recorded by the gateway."""
        return {"traces": trace_store.list_recent(limit=50)}

    @app.get("/v1/traces/{request_id}")
    async def get_trace(
        request_id: str,
        trace_store: InMemoryTraceStore = Depends(get_trace_store),
    ):
        """Return the 6-milestone lifecycle trace for a specific request ID."""
        trace = trace_store.get(request_id)
        if trace is None:
            raise HTTPException(status_code=404, detail=f"Trace {request_id} not found.")
        return trace.to_dict()

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request,
        request_payload: ChatCompletionRequest,
        router: RoutingPolicy = Depends(get_router),
        trace_store: InMemoryTraceStore = Depends(get_trace_store),
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

        # 1. Create request trace and mark arrival
        trace = RequestTrace(request_id=request_id, model=model_name)
        trace.mark_arrival()

        # 2. Route request via injected RoutingPolicy and mark routing_start / routing_end
        target_worker = router.select_worker(trace)
        worker_id = trace.worker_id
        trace_store.save(trace)

        payload = request_payload.model_dump(exclude_none=True)

        log_structured_event(
            event="chat_stream_started",
            request_id=request_id,
            client_ip=client_ip,
            model=model_name,
            worker_id=worker_id,
            status="started",
            extra={"arrival": trace.arrival, "routing_ms": trace.routing_ms},
        )

        return StreamingResponse(
            safe_event_generator(
                target_worker,
                payload,
                request_id,
                client_ip,
                trace=trace,
                trace_store=trace_store,
            ),
            media_type="text/event-stream",
            headers={
                "X-Request-ID": request_id,
                "X-Worker-ID": worker_id,
            },
        )

    return app


app = create_app()
