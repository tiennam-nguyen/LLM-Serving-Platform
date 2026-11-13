# System Architecture & Week 4 Snapshot

This document records the architecture principles, active **Week 4 implementation snapshot**, **SOLID Worker Pool Factory & Unified Verification Architecture**, and target **Week 5–12 distributed expansion topology** for the **Adaptive LLM Serving Platform**.

---

## 1. Architectural Principles (SOLID & Clean Boundaries)

1. **Modular Monolith First**:
   The backend is structured as an installable `src`-layout Python package (`llm_serving_platform`) with explicit module boundaries (`gateway`, `routing`, `workers`, `telemetry`, `control`, `benchmark`, `common`). All internal, script, and test imports use the canonical `llm_serving_platform.*` namespace—runtime and operational scripts never import from `tests/`.

2. **Decoupled Data Plane and Control Plane**:
   - The **data plane** (`gateway`, `routing`, `workers`, `telemetry`) processes live HTTP/SSE inference streams strictly in memory without synchronous database or network health-probe calls on the critical path.
   - The **control plane** (`control/registry.py`, `control/health.py`, `control/api.py`) maintains thread-safe worker metadata (`WorkerRegistry`), executes asynchronous concurrent health probes (`HealthMonitor.check_once()`) on a background schedule, and exposes read-only topology endpoints (`GET /control/v1/workers` and `GET /control/v1/workers/{worker_id}`).

3. **Dependency Inversion for Routing (`RoutingPolicy` ABC & `RoundRobinRouter`)**:
   HTTP route handlers in `gateway/main.py` depend on the abstract `RoutingPolicy` interface (`Depends(get_router)`), never instantiating concrete routers inside request handlers.
   - In **Week 3**, `SingleWorkerRouter` served as the single-worker walking skeleton.
   - In **Week 4**, `RoundRobinRouter(registry)` implements `RoutingPolicy`, consuming cached `HEALTHY` worker instances from `WorkerRegistry.healthy_workers()`, cycling sequentially across healthy candidates, excluding `UNHEALTHY`/`STARTING` workers automatically, and raising `WorkerUnavailableError` (HTTP 503) when zero healthy workers remain.
   - In **Week 6**, `LeastLoadedRouter` swaps into `create_router()` without modifying the HTTP controller.

4. **Runtime-Agnostic Worker Pool Factory (`InferenceWorker` ABC + `create_worker` / `populate_registry`)**:
   Serving engines sit behind `InferenceWorker` (`stream_chat`, `check_health`). `workers/factory.py` decouples the Gateway and Registry from concrete worker implementations:
   - **Live GPU Mode (Default, `no_gpu=False`)**: Instantiates one or more `VLLMWorkerAdapter` instances from `WORKER_ENDPOINTS` (or single `VLLM_BASE_URL` / `WORKER_ID`). App construction allocates no network client; FastAPI lifespan creates exactly one shared `httpx.AsyncClient`, attaches it to all live adapters, and closes it on shutdown. The adapter encapsulates vLLM's OpenAI-compatible HTTP/SSE wire protocol and maps network/timeout failures into typed `WorkerTimeoutError` (504) and `WorkerUnavailableError` (503) exceptions.
   - **No-GPU Mode (`--no-gpu` / `NO_GPU=1` / `WORKER_MODE=mock`)**: Instantiates a configurable pool of `MockInferenceWorker` instances (`MOCK_WORKER_COUNT=2` by default: `worker-mock-0`, `worker-mock-1`) directly inside the FastAPI Gateway process. This enables deterministic multi-worker Round-Robin CI testing AND full interactive browser testing (`http://127.0.0.1:3000` and `http://127.0.0.1:3000/workers` -> `:8080`) on CPU-only laptops without a physical GPU.
   - **Clean Dependency Packaging (`pyproject.toml`)**: Core Data-Plane & Control-Plane Gateway dependencies (`fastapi`, `httpx`, `uvicorn`, `pydantic`) are separated from heavy CUDA/worker dependencies (`[project.optional-dependencies].gpu` containing `vllm`, which declares its supported Transformers range).

5. **Open-Closed Cumulative Verification Pipeline (`infra/scripts/verify.py`)**:
   All quality stages (`lint`, `unit`, `integration`, `gates`, `web`) and cumulative weekly exit gates (`w1`, `w2`, `w3`, `w4`, and future `w5+`) are orchestrated through a single declarative registry (`MILESTONE_GATES`) in `infra/scripts/verify.py`. Future weeks add new gate entries without deleting or breaking prior weekly gates, and `.github/workflows/ci.yml` invokes the same entrypoint in explicit CPU/mock stages. Hosted CI intentionally does not claim live-GPU evidence; live hardware checks are executed separately on the available GPU topology.

6. **Honest Telemetry & Metric Semantics**:
   Client-observed timings (`FIRST CONTENT`, `E2E DURATION` in `apps/web`) are kept distinct from server-observed monotonic trace milestones (`RequestTrace` in `telemetry/trace.py`). Because a single SSE `data:` frame is not guaranteed to equal one model token, chunk spacing is published honestly as `inter_content_chunk_ms`, while `tpot_ms` remains `None` unless true `completion_tokens > 1` is reported by the runtime.

---

## 2. Week 4 Active Topology Snapshot (Current Implementation)

In Week 4, the platform operates a multi-worker Control Plane (`WorkerRegistry` + `HealthMonitor` + `/control/v1/workers`) integrated with the Data Plane (`RoundRobinRouter`) and the Next.js `/workers` Topology UI, supporting both **Live GPU (default)** and **No-GPU (`--no-gpu` / `NO_GPU=1`)** execution across CLI, CI, and interactive browser sessions.

### 2.1 Week 4 Request, Control-Plane Health & Trace Flow

```text
+-----------------------------------------------------------------------------------+
|                        Browser: Next.js Web UI (apps/web)                         |
|  - Playground (/)      : Multi-turn SSE chat, Abort/Retry, RequestInspector       |
|  - Workers (/workers)  : Control-Plane Topology View, Manual Refresh, Status Tags |
+-----------------+-----------------------------------------------+-----------------+
                  |                                               |
   1. POST /v1/chat/completions                    4. GET /control/v1/workers
      (SSE stream + X-Request-ID, X-Worker-ID)        (Read-only WorkerRecord list)
   3. GET /v1/traces/{request_id}                                 |
                  |                                               |
                  v                                               v
+-----------------------------------------------------------------------------------+
|                    FastAPI Gateway (gateway/main.py)                              |
|  - CORSMiddleware (outermost) + request_id_middleware                             |
|  - App factory: populate_registry(); lifespan: attach one client -> health loop   |
+-----------------+-------------------------------+---------------+-----------------+
                  |                               |               |
   2a. select_worker(trace)        2b. save/get   |               | Read snapshots
                  v                    trace      v               v
+----------------------------------+  +---------------+  +--------------------------+
|   RoundRobinRouter (routing/)    |  | InMemoryTrace |  | WorkerRegistry (control/)|
|  - Reads cached healthy_workers()|  | Store (FIFO   |  | - WorkerRecord snapshots |
|  - Cycles cursor thread-safely   |  | max_size=500) |  | - STARTING / HEALTHY /   |
|  - 0 healthy -> 503 Unavailable  |  +---------------+  |   UNHEALTHY states       |
+-----------------+----------------+                     +------------+-------------+
                  |                                                   ^
                  | Stream SSE from selected healthy worker           | Background
                  v                                                   | check_once()
+---------------------------------------------------------------------+-------------+
|          Worker Pool Factory: create_worker_pool / populate_registry              |
|  - Default (no_gpu=False) : VLLMWorkerAdapter pool (worker-vllm-0, ...)           |
|  - No-GPU  (no_gpu=True)  : MockInferenceWorker pool (worker-mock-0, worker-mock-1)|
+-----------------------------------------------------------------------------------+
```

### 2.2 Worker Health State Machine (`WorkerStatus`)

| Status | Trigger / Condition | Routing Eligibility (`healthy_workers()`) |
| :--- | :--- | :--- |
| `STARTING` | Initial state upon `registry.register(worker, endpoint, model)` (`last_checked_at = None`). | **Excluded** (not routed until first probe succeeds). |
| `HEALTHY` | `HealthMonitor.check_once()` receives `True` from `worker.check_health()` within `timeout_s`. | **Included** in `RoundRobinRouter` candidate set. |
| `UNHEALTHY` | `worker.check_health()` returns `False`, raises an exception, or exceeds `timeout_s`. | **Excluded** immediately; if all workers are `UNHEALTHY`, router raises `WorkerUnavailableError` (503). |

---

## 3. Module Boundaries & Week 4 Implementation Status

| Module | Boundary Role | Week 4 Implementation Status |
| :--- | :--- | :--- |
| `gateway` | Data-plane HTTP/SSE entrypoint & lifespan wiring | **Active**: `main.py`, `schemas.py`, `streaming.py` (`/health`, `/v1/chat/completions`, `/v1/traces`, mounts `/control/v1/workers`, manages `HealthMonitor` lifespan loop & 503 empty-pool handling). |
| `control` | Worker registry, health monitor & read-only API | **Active**: `WorkerRegistry` & `WorkerRecord` (`registry.py`), concurrent `HealthMonitor` (`health.py`), `create_control_router` (`api.py`). |
| `routing` | Pluggable worker selection policy | **Active**: `RoutingPolicy` ABC, `RoundRobinRouter`, and `SingleWorkerRouter` (`router.py`). |
| `workers` | Serving runtime abstraction & pool factory | **Active**: `InferenceWorker` ABC (`base.py`), `VLLMWorkerAdapter` (`vllm_adapter.py`), `MockInferenceWorker` (`mock_adapter.py`), `create_worker`, `populate_registry` (`factory.py`). |
| `telemetry` | Per-request tracing & metrics | **Active**: `RequestTrace` (6 milestones + honest `inter_content_chunk_ms`/`tpot_ms`) & `InMemoryTraceStore` (`trace.py`). |
| `common` | Config & structured logging | **Active**: `GatewayConfig` (`no_gpu`, `worker_endpoints`, `mock_worker_count`, `routing_policy`, `health_check_interval_s`, `health_check_timeout_s`) (`config.py`), JSON logger (`logging.py`). |
| `apps/web` | Operator console, playground & topology UI | **Active**: Next.js 16 / React 19 Playground (`/`) + Control-Plane Workers Topology view (`/workers`). |
| `infra/scripts` | Unified verification & interactive dev stack | **Active**: `verify.py` (cumulative gates `w1`–`w4`), `verify_week4_e2e.py`, `dev_stack.py` (multi-worker interactive stack). |
| `benchmark` | Reproducible workload runner | **Planned (Week 8)**: controlled replay/benchmark suite after Week 7 observability and experiment artifacts. |
