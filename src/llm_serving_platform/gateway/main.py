"""FastAPI API Gateway entrypoint for LLM Serving Platform."""

import asyncio
import contextlib
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from llm_serving_platform.common.config import GatewayConfig
from llm_serving_platform.common.logging import log_structured_event
from llm_serving_platform.control.api import create_control_router
from llm_serving_platform.control.health import HealthMonitor
from llm_serving_platform.control.registry import WorkerRegistry
from llm_serving_platform.gateway.schemas import ChatCompletionRequest
from llm_serving_platform.gateway.streaming import safe_event_generator
from llm_serving_platform.routing.router import (
    RoundRobinRouter,
    RoutingPolicy,
    SingleWorkerRouter,
)
from llm_serving_platform.telemetry.trace import InMemoryTraceStore, RequestTrace
from llm_serving_platform.workers.base import InferenceWorker, WorkerUnavailableError
from llm_serving_platform.workers.factory import create_worker, populate_registry
from llm_serving_platform.workers.vllm_adapter import VLLMWorkerAdapter


def _build_http_client(config: GatewayConfig) -> httpx.AsyncClient:
    limits = httpx.Limits(max_keepalive_connections=50, max_connections=200)
    timeout = httpx.Timeout(config.worker_timeout_s, connect=5.0)
    return httpx.AsyncClient(limits=limits, timeout=timeout)


def _attach_http_client(registry: WorkerRegistry, client: httpx.AsyncClient) -> None:
    for _, worker in registry.probe_targets():
        if isinstance(worker, VLLMWorkerAdapter):
            worker.client = client


async def _periodic_health_loop(monitor: HealthMonitor, interval_s: float) -> None:
    while True:
        await asyncio.sleep(interval_s)
        await monitor.check_once()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the lifecycle of the shared httpx client, health monitor loop, and trace store."""
    config: GatewayConfig = app.state.config
    client: httpx.AsyncClient | None = None
    if not config.no_gpu:
        client = _build_http_client(config)
        app.state.client = client
        _attach_http_client(app.state.registry, client)

    await app.state.health_monitor.check_once()
    app.state.initial_health_checked = True

    health_task: asyncio.Task[None] | None = None
    if config.health_check_interval_s > 0:
        health_task = asyncio.create_task(
            _periodic_health_loop(app.state.health_monitor, config.health_check_interval_s)
        )
    app.state.health_task = health_task

    try:
        yield
    finally:
        if health_task is not None:
            health_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await health_task
        if client is not None:
            await client.aclose()
            app.state.client = None


async def get_worker(request: Request) -> InferenceWorker:
    """Async dependency provider for the primary inference worker (backward compatible)."""
    return request.app.state.worker


async def get_router(
    request: Request,
    worker: InferenceWorker = Depends(get_worker),
) -> RoutingPolicy:
    """Dependency provider for the routing policy, decoupled from HTTP controller logic."""
    if get_worker in request.app.dependency_overrides:
        return SingleWorkerRouter(worker)

    if not getattr(request.app.state, "initial_health_checked", False):
        await request.app.state.health_monitor.check_once()
        request.app.state.initial_health_checked = True

    return request.app.state.router


async def get_trace_store(request: Request) -> InMemoryTraceStore:
    """Async dependency provider for the in-memory request trace store."""
    return request.app.state.trace_store


def create_app(
    config: GatewayConfig | None = None,
    registry: WorkerRegistry | None = None,
) -> FastAPI:
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
    # Network resources are lifespan-owned. App construction must not allocate an AsyncClient.
    app.state.client = None

    active_registry = registry if registry is not None else WorkerRegistry()
    if registry is None:
        workers = populate_registry(active_registry, cfg, client=None)
        app.state.worker = workers[0]
        app.state.initial_health_checked = cfg.no_gpu
    else:
        targets = active_registry.probe_targets()
        app.state.worker = (
            targets[0][1] if targets else create_worker(config=cfg, client=app.state.client)
        )
        app.state.initial_health_checked = True

    app.state.registry = active_registry
    app.state.health_monitor = HealthMonitor(
        active_registry,
        timeout_s=cfg.health_check_timeout_s,
    )
    app.state.router = (
        SingleWorkerRouter(app.state.worker)
        if cfg.routing_policy in {"single", "single_worker"}
        else RoundRobinRouter(active_registry)
    )
    app.include_router(create_control_router(active_registry))

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
    async def health(
        request: Request,
        worker: InferenceWorker = Depends(get_worker),
    ):
        """Check the health status of the Gateway and backend worker pool."""
        if get_worker in request.app.dependency_overrides:
            worker_id = getattr(worker, "worker_id", "worker-vllm-0")
            is_healthy = await worker.check_health()
        else:
            await request.app.state.health_monitor.check_once()
            request.app.state.initial_health_checked = True
            healthy = request.app.state.registry.healthy_workers()
            records = request.app.state.registry.list_workers()
            is_healthy = bool(healthy)
            worker_id = (
                getattr(healthy[0], "worker_id", "worker-vllm-0")
                if healthy
                else (records[0].worker_id if records else "none")
            )

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
        try:
            target_worker = router.select_worker(trace)
        except WorkerUnavailableError as exc:
            trace.mark_last_token(status="error", error=str(exc))
            trace_store.save(trace)
            log_structured_event(
                event="routing_no_healthy_worker",
                request_id=request_id,
                client_ip=client_ip,
                model=model_name,
                worker_id="none",
                status="error",
                extra={"error": str(exc), "code": exc.status_code, **trace.to_dict()},
            )
            return JSONResponse(
                status_code=exc.status_code,
                content={"error": {"message": str(exc), "code": exc.status_code}},
                headers={"X-Request-ID": request_id, "X-Worker-ID": "none"},
            )

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
