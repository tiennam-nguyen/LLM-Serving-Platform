# Adaptive LLM Serving Platform for Observable Multi-GPU Inference

## 1. Overview

The **Adaptive LLM Serving Platform** is an observable, reproducible, multi-worker LLM inference serving platform designed to manage, trace, and route LLM workloads across GPU nodes.

The system follows a **modular monolith** architecture with strict separation of concerns (`SOLID`) between the **data-plane HTTP/SSE gateway**, **pluggable routing policies**, **runtime-agnostic worker adapters**, **6-milestone per-request telemetry**, and the **Next.js operator console (`apps/web`)**.

---

## 2. Current Status: Week 3 Completed (Gate A Ready)

> [!NOTE]
> **Status**: Week 3 Complete — End-to-End Web Walking Skeleton (`Browser → Gateway → Worker → GPU`).
>
> - **Week 1 (Single-GPU vLLM Baseline)**: Completed (`Qwen/Qwen2.5-0.5B-Instruct` on CUDA `float16`, launch scripts, latency measurement).
> - **Week 2 (Data-Plane Inference Gateway)**: Completed (OpenAI-compatible `POST /v1/chat/completions` SSE proxy, `InferenceWorker` / `VLLMWorkerAdapter`, `X-Request-ID`, structured JSON logging, timeout/error propagation).
> - **Week 3 (Minimal End-to-End Web Flow & Request Trace)**: Completed (Next.js Instrumentation Deck UI, multi-turn streaming, abort/retry, `X-Worker-ID`, 6-milestone `RequestTrace` exposed via `GET /v1/traces/{request_id}`, and dual-mode `--no-gpu` / live GPU verification).
> - **Next Milestone**: Week 4 — Worker Registry, Health State Machine (`HEALTHY`/`DEGRADED`/`UNHEALTHY`/`OFFLINE`), and Round-Robin `RoutingPolicy`.

---

## 3. Semester Engineering Progression

```text
[x] Week 1: 1 GPU / 1 worker (vLLM baseline)
    ↓
[x] Week 2: Unified data-plane gateway + VLLMWorkerAdapter
    ↓
[x] Week 3: Web E2E operator/chat UI + 6-milestone RequestTrace
    ↓
[ ] Week 4: Worker registry + health state machine + Round-Robin routing
    ↓
[ ] Week 5: Stabilization & integration buffer (Gate A lock)
    ↓
[ ] Week 6: LAN PoC (multi-host networking & failure recovery)
    ↓
[ ] Week 7: 2+ independent GPU replicas + Least-Loaded routing policy
    ↓
[ ] Week 8–9: Observability + reproducible benchmark runner
    ↓
[ ] Week 11–12: Final routing experiments & feature freeze
```

---

## 4. Repository Layout

```text
.
├── src/
│   └── llm_serving_platform/       # Root Python package (modular monolith)
│       ├── __init__.py
│       ├── __main__.py             # Bootstrap verification CLI entrypoint
│       ├── gateway/                # FastAPI data-plane HTTP/SSE entrypoint (main, schemas, streaming)
│       ├── routing/                # RoutingPolicy ABC & SingleWorkerRouter (Week 4: RoundRobin)
│       ├── workers/                # InferenceWorker ABC & VLLMWorkerAdapter
│       ├── telemetry/              # 6-milestone RequestTrace & InMemoryTraceStore
│       ├── control/                # Control plane boundary (Week 4: WorkerRegistry & HealthMonitor)
│       ├── benchmark/              # Workload replay & benchmark orchestration boundary
│       └── common/                 # 12-Factor GatewayConfig (.env loader) & structured JSON logging
│
├── apps/
│   └── web/                        # Next.js / React / TypeScript Instrumentation Deck & Chat Playground
│
├── infra/
│   ├── compose/                    # Hardened Dockerfiles, seccomp profile & Docker Compose (gpu / no-gpu)
│   ├── monitoring/                 # Monitoring and metrics configuration
│   └── scripts/                    # Launch scripts & dual-mode verification (verify_week3_e2e.py, etc.)
│
├── tests/
│   ├── conftest.py                 # Pytest --no-gpu and --prompt fixtures
│   ├── mocks.py                    # In-memory MockInferenceWorker
│   ├── unit/                       # Unit tests (validation, health, CORS, traces, package structure)
│   └── integration/                # Integration & dual-mode E2E tests
│
├── docs/
│   ├── architecture.md             # Week 3 Architecture Snapshot & component boundaries
│   ├── report_outline_week3.md     # Week 3 Report Outline & milestone evidence summary
│   ├── adr/                        # Architecture Decision Records
│   ├── api/                        # API specifications & interface contracts
│   ├── experiments/                # Experiment methodology & reproducibility guides
│   └── runbooks/                   # Operational runbooks (WSL2 + vLLM GPU setup)
│
├── .github/workflows/ci.yml        # GitHub Actions CI (CPU / --no-gpu backend + frontend suites)
├── pyproject.toml                  # Python package configuration, dependencies & Ruff/Pytest settings
└── README.md                       # Project documentation entry point
```

---

## 5. Local Setup & Verification Commands

### 5.1 Python Backend Setup

```bash
# Create and activate virtual environment (Python 3.11 or 3.12)
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1

# Install package in editable mode (use .[ci] on CPU-only machines without CUDA/vLLM)
python -m pip install -e ".[dev]"
```

### 5.2 Dual-Mode Verification (GPU vs. `--no-gpu` and Custom Prompt `--prompt`)

All verification scripts support **two modes**:
1. **Default Mode (Live GPU)**: Connects to the live Gateway (`http://127.0.0.1:8080`) and vLLM GPU worker (`http://127.0.0.1:8000`), sending a single default prompt (or a custom prompt via `--prompt "..."`).
2. **No-GPU Mode (`--no-gpu`)**: Uses the in-process `MockInferenceWorker` so CI pipelines and laptops without an NVIDIA GPU can verify the full Gateway, Routing, SSE Streaming, and 6-milestone Trace pipeline.

```bash
# 1. Run pytest in CPU/CI mode (no GPU required)
python -m pytest --no-gpu

# 2. Run pytest with a custom prompt
python -m pytest --no-gpu --prompt "Giải thích KV-cache trong 2 câu."

# 3. Run Week 3 E2E verification script in --no-gpu mode
python infra/scripts/verify_week3_e2e.py --no-gpu
python infra/scripts/verify_week3_e2e.py --no-gpu --prompt "Giải thích cơ chế Continuous Batching."

# 4. Run Week 3 E2E verification script on Live GPU (when vLLM + Gateway are running)
python infra/scripts/verify_week3_e2e.py
python infra/scripts/verify_week3_e2e.py --prompt "Giải thích ngắn gọn PagedAttention trong vLLM."

# 5. Run Ruff linter and formatter checks
python -m ruff check .
python -m ruff format --check .
```

### 5.3 Frontend (`apps/web`) Setup & Verification

```bash
cd apps/web
npm ci
npm run lint
npm run typecheck
npm test
npm run build
npm run test:browser
```

---

## 6. Next Milestone: Week 4

**Week 4 Target**: Multi-Worker Registry, Health Monitoring, and Round-Robin Routing.
1. Implement `WorkerRegistry` (`worker_id`, `node_id`, `endpoint`, `model`, `gpu_id`, `status`, `last_seen`).
2. Add active health checking with state transitions (`HEALTHY`, `DEGRADED`, `UNHEALTHY`, `OFFLINE`).
3. Implement `RoundRobinRouter(RoutingPolicy)` skipping unhealthy workers.
4. Expose Worker/Topology status view in the Web UI.
