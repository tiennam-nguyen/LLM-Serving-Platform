# System Architecture & Week 3 Snapshot

This document records the architecture principles, active **Week 3 implementation snapshot**, and target **Week 4+ multi-worker expansion topology** for the **Adaptive LLM Serving Platform**.

---

## 1. Architectural Principles (SOLID & Clean Boundaries)

1. **Modular Monolith First**:
   The backend is structured as an installable `src`-layout Python package (`llm_serving_platform`) with explicit module boundaries (`gateway`, `routing`, `workers`, `telemetry`, `control`, `benchmark`, `common`). All internal and test imports use the canonical `llm_serving_platform.*` namespace.

2. **Decoupled Data Plane and Control Plane**:
   The **data plane** (`gateway`, `routing`, `workers`, `telemetry`) processes live HTTP/SSE inference streams strictly in memory without synchronous database calls on the critical path.

3. **Dependency Inversion for Routing (`RoutingPolicy` ABC)**:
   HTTP route handlers in `gateway/main.py` depend on the abstract `RoutingPolicy` interface (`Depends(get_router)`), never instantiating concrete routers inside request handlers. In Week 3, `SingleWorkerRouter` implements `RoutingPolicy`; in Week 4 and Week 7, `RoundRobinRouter` and `LeastLoadedRouter` swap in without modifying the HTTP controller.

4. **Runtime-Agnostic Worker Adapters (`InferenceWorker` ABC)**:
   Serving engines sit behind `InferenceWorker` (`stream_chat`, `check_health`). `VLLMWorkerAdapter` encapsulates vLLM's OpenAI-compatible HTTP/SSE wire protocol and maps `httpx` network/timeout failures into typed `WorkerTimeoutError` (504) and `WorkerUnavailableError` (503) exceptions. `MockInferenceWorker` implements the exact same contract for deterministic `--no-gpu` testing.

5. **Honest Telemetry & Metric Semantics**:
    client-observed timings (`FIRST CONTENT`, `E2E DURATION` in `apps/web`) are kept distinct from server-observed monotonic trace milestones (`RequestTrace` in `telemetry/trace.py`). Because a single SSE `data:` frame is not guaranteed to equal one model token, chunk spacing is published honestly as `inter_content_chunk_ms`, while `tpot_ms` remains `None` unless true `completion_tokens > 1` is reported by the runtime.

---

## 2. Week 3 Active Topology Snapshot (Current Implementation)

In Week 3, the platform operates an end-to-end single-worker walking skeleton from the browser to the GPU worker with full request correlation (`X-Request-ID`, `X-Worker-ID`) and a 6-milestone server trace store.

### 2.1 Week 3 Request & Trace Flow

```text
+-------------------------------------------------------------------------+
|                   Browser: Next.js Web UI (apps/web)                    |
|  - PlaygroundDeck / useChatSession (multi-turn, AbortController, retry) |
|  - SSEParser (partial chunks, CRLF, multiline SSE, explicit usage)      |
|  - RequestInspector (client FIRST CONTENT/E2E + server trace display)   |
+------------------------------------+------------------------------------+
                                     |
        1. POST /v1/chat/completions |  3. GET /v1/traces/{request_id}
           (SSE stream + CORS headers|     (fetches 6-milestone server trace
            X-Request-ID, X-Worker-ID|      after stream completion)
                                     v
+-------------------------------------------------------------------------+
|               FastAPI Data-Plane Gateway (gateway/main.py)              |
|  - CORSMiddleware (outermost) + request_id_middleware                   |
|  - ChatCompletionRequest validation (gateway/schemas.py)                |
|  - safe_event_generator SSE orchestrator (gateway/streaming.py)         |
+-----------------+----------------------------------+--------------------+
                  |                                  |
                  | 2a. select_worker(trace)         | 2b. save/get trace
                  v                                  v
+----------------------------------+  +-----------------------------------+
|      RoutingPolicy (routing/)    |  |   InMemoryTraceStore (telemetry/) |
|  - Week 3: SingleWorkerRouter    |  |  - Bounded FIFO (max_size=500)    |
|  - Marks routing_start &         |  |  - Stores RequestTrace:           |
|    routing_end(worker_id)        |  |    arrival -> routing_start ->    |
+-----------------+----------------+  |    routing_end -> worker_start -> |
                  |                   |    first_token -> last_token      |
                  |                   +-----------------------------------+
                  v
+-------------------------------------------------------------------------+
|                   InferenceWorker Adapter (workers/)                    |
|  - Live GPU Mode : VLLMWorkerAdapter -> http://127.0.0.1:8000 (vLLM)    |
|  - No-GPU Mode   : MockInferenceWorker (in-process deterministic SSE)   |
+------------------------------------+------------------------------------+
                                     |
                                     v (Live GPU Mode)
+-------------------------------------------------------------------------+
|            vLLM OpenAI API Server on NVIDIA GPU (float16)               |
|            Model: Qwen/Qwen2.5-0.5B-Instruct (Continuous Batching)      |
+-------------------------------------------------------------------------+
```

### 2.2 Week 3 Trace Milestones (`RequestTrace`)

Every streaming request records 6 UTC ISO-8601 timestamps alongside internal `time.perf_counter()` monotonic timestamps:

| Milestone | Recorded By | Description |
| :--- | :--- | :--- |
| `arrival` | `gateway/main.py` | Request validated and accepted at Gateway entrypoint. |
| `routing_start` | `routing/router.py` | Router begins evaluating candidate worker(s). |
| `routing_end` | `routing/router.py` | Target `worker_id` selected (`routing_ms = routing_end - routing_start`). |
| `worker_start` | `gateway/streaming.py` | `safe_event_generator` opens upstream stream to `InferenceWorker`. |
| `first_token` | `gateway/streaming.py` | First non-empty assistant content chunk received (`ttft_ms = first_token - arrival`). |
| `last_token` | `gateway/streaming.py` | Stream reaches terminal state (`completed`, `cancelled`, or `error`; `e2e_ms = last_token - arrival`). |

---

## 3. Target Multi-Worker Topology (Week 4 – Week 12 Roadmap)

Starting in Week 4 (`WorkerRegistry` + `RoundRobinRouter`) and Week 7 (`LeastLoadedRouter` across 2+ GPU replicas), the `RoutingPolicy` boundary expands from a single worker to a dynamic pool of health-checked workers:

```text
+-----------------------+       +-------------------------+
|   Web UI / Client     |       | Benchmark Runner Client |
+-----------+-----------+       +------------+------------+
            |                                |
            +----------------+---------------+
                             |
                             v
           +-----------------------------------+
           |        Serving Gateway            | (Data Plane: Protocol termination,
           +-----------------+-----------------+  streaming, validation, tracing)
                             |
                             v
           +-----------------------------------+
           |     RoutingPolicy Engine          | (Week 4: RoundRobinRouter
           |  + WorkerRegistry Health Snapshot |  Week 7: LeastLoadedRouter)
           +-----------------+-----------------+
                             |
            +----------------+----------------+
            |                                 |
            v                                 v
+-----------------------+         +-----------------------+
|  VLLMWorkerAdapter #1 |         |  VLLMWorkerAdapter #2 |
+-----------+-----------+         +-----------+-----------+
            |                                 |
            v                                 v
+-----------------------+         +-----------------------+
|   vLLM Replica #1     |         |   vLLM Replica #2     |
+-----------------------+         +-----------------------+
```

---

## 4. Module Boundaries & Responsibilities

| Module | Boundary Role | Week 3 Implementation Status |
| :--- | :--- | :--- |
| `gateway` | Data-plane HTTP/SSE entrypoint | **Active**: `main.py`, `schemas.py`, `streaming.py` (`/health`, `/v1/chat/completions`, `/v1/traces`). |
| `routing` | Worker selection policy | **Active**: `RoutingPolicy` ABC + `SingleWorkerRouter` (`router.py`). |
| `workers` | Serving runtime abstraction | **Active**: `InferenceWorker` ABC (`base.py`), `VLLMWorkerAdapter` (`vllm_adapter.py`), `MockInferenceWorker` (`tests/mocks.py`). |
| `telemetry` | Per-request tracing & metrics | **Active**: `RequestTrace` (6 milestones + honest `inter_content_chunk_ms`/`tpot_ms`) & `InMemoryTraceStore` (`trace.py`). |
| `common` | Config & structured logging | **Active**: `GatewayConfig` + `.env` loader (`config.py`), JSON logger (`logging.py`). |
| `control` | Cluster & worker registry | **Planned (Week 4)**: `WorkerRegistry` & health state machine (`HEALTHY`/`DEGRADED`/`UNHEALTHY`/`OFFLINE`). |
| `benchmark` | Reproducible workload runner | **Planned (Week 8–9)**: Trace replay, Poisson/burst arrival generators, percentile aggregator. |
| `apps/web` | Operator console & playground | **Active**: Next.js 16 / React 19 UI with SSE streaming, abort/retry, and `RequestInspector`. |
