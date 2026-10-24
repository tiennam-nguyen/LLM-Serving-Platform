# Adaptive LLM Serving Platform for Observable Multi-GPU Inference

## 1. Overview

The **Adaptive LLM Serving Platform** is an observable, reproducible, multi-worker LLM inference serving platform designed to manage, trace, and route LLM workloads across GPU nodes.

The system follows a **modular monolith** architecture with strict separation of concerns (`SOLID`) between the **data-plane HTTP/SSE gateway**, **pluggable routing policies**, **runtime-agnostic worker adapters (`create_worker` factory)**, **6-milestone per-request telemetry**, **cumulative Open-Closed verification gates (`infra/scripts/verify.py`)**, and the **Next.js operator console (`apps/web`)**.

---

## 2. Current Status: Week 3 Completed (Gate A Ready)

> [!NOTE]
> **Status**: Week 3 Complete — End-to-End Web Walking Skeleton (`Browser → Gateway → Worker → GPU / No-GPU`).
>
> - **Week 1 (Single-GPU vLLM Baseline)**: Completed (`Qwen/Qwen2.5-0.5B-Instruct` on CUDA `float16`, launch scripts, latency measurement).
> - **Week 2 (Data-Plane Inference Gateway)**: Completed (OpenAI-compatible `POST /v1/chat/completions` SSE proxy, `InferenceWorker` / `VLLMWorkerAdapter`, `X-Request-ID`, structured JSON logging, timeout/error propagation).
> - **Week 3 (Minimal End-to-End Web Flow, Request Trace & Unified Verification)**: Completed (Next.js Instrumentation Deck UI, multi-turn streaming, abort/retry, `X-Worker-ID`, 6-milestone `RequestTrace` via `GET /v1/traces/{request_id}`, `create_worker` factory, unified `verify.py` runner, and `dev_stack.py` interactive browser launcher).
> - **Next Milestone**: Week 4 — Worker Registry, Health State Machine (`HEALTHY`/`DEGRADED`/`UNHEALTHY`/`OFFLINE`), and Round-Robin `RoutingPolicy`.

---

## 3. Semester Engineering Progression

```text
[x] Week 1: 1 GPU / 1 worker (vLLM baseline)
    ↓
[x] Week 2: Unified data-plane gateway + VLLMWorkerAdapter
    ↓
[x] Week 3: Web E2E operator/chat UI + 6-milestone RequestTrace + Unified Verify/Stack CLI
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
│       ├── workers/                # InferenceWorker ABC, VLLMWorkerAdapter, MockInferenceWorker, factory
│       ├── telemetry/              # 6-milestone RequestTrace & InMemoryTraceStore
│       ├── control/                # Control plane boundary (Week 4: WorkerRegistry & HealthMonitor)
│       ├── benchmark/              # Workload replay & benchmark orchestration boundary
│       └── common/                 # 12-Factor GatewayConfig (.env + NO_GPU loader) & JSON logging
│
├── apps/
│   └── web/                        # Next.js / React / TypeScript Instrumentation Deck & Chat Playground
│
├── infra/
│   ├── compose/                    # Hardened Dockerfiles & Docker Compose (GPU profile & NO_GPU=1)
│   ├── monitoring/                 # Monitoring and metrics configuration
│   └── scripts/                    # Unified verify.py, dev_stack.py, and milestone scripts
│
├── tests/
│   ├── conftest.py                 # Pytest --no-gpu and --prompt fixtures
│   ├── mocks.py                    # Re-exports canonical MockInferenceWorker
│   ├── unit/                       # Unit tests (validation, health, CORS, traces, factory, package)
│   └── integration/                # Integration & dual-mode E2E tests
│
├── docs/
│   ├── architecture.md             # Week 3 Architecture Snapshot & SOLID boundaries
│   ├── report_outline_week3.md     # Week 3 Report Outline & milestone evidence summary
│   └── runbooks/setup_wsl.md       # Operational runbook (WSL2 + vLLM GPU setup)
│
├── .github/workflows/ci.yml        # Single-command GitHub Actions CI calling verify.py --no-gpu
├── pyproject.toml                  # Python package configuration, dependencies & Ruff/Pytest settings
└── README.md                       # Project documentation entry point
```

---

## 5. Setup & Unified Testing Guide (GPU Default vs. `--no-gpu`)

### 5.1 Environment Setup

```bash
# 1. Python Backend Setup (Python 3.11 or 3.12)
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --no-deps -e .
python -m pip install ".[ci]"   # Or ".[dev]" on GPU machines with CUDA/vLLM

# 2. Frontend Setup
cd apps/web
npm ci
npx playwright install chromium
cd ../..
```

### 5.2 Mode 1: Automated E2E or Partial Verification (`infra/scripts/verify.py`)

`infra/scripts/verify.py` is the **single source of truth** for all testing (Lint, Unit, Integration, Cumulative Weekly Gates `w1` → `w2` → `w3`, and Frontend/Playwright).
- **Default**: Runs in **Live GPU mode** (`http://127.0.0.1:8080` → vLLM `:8000`).
- **`--no-gpu`**: Runs in **No-GPU mode** using `MockInferenceWorker`.

```bash
# A. Full End-to-End Verification (All stages + Week 1-3 Gates + Web E2E)
python infra/scripts/verify.py                # Default: Live GPU
python infra/scripts/verify.py --no-gpu       # No-GPU (CPU / CI mode)

# B. Partial / Component Verification (both GPU default and --no-gpu supported)
python infra/scripts/verify.py --target lint --no-gpu
python infra/scripts/verify.py --target unit --no-gpu
python infra/scripts/verify.py --target integration --no-gpu
python infra/scripts/verify.py --target w1 --no-gpu      # Week 1 Gate only (or omit --no-gpu for GPU)
python infra/scripts/verify.py --target w2 --no-gpu      # Week 2 Gate only
python infra/scripts/verify.py --target w3 --no-gpu      # Week 3 Gate only
python infra/scripts/verify.py --target gates --no-gpu   # All cumulative weekly gates (W1 -> W3)
python infra/scripts/verify.py --target web --no-gpu     # Frontend lint, typecheck, vitest, build, Playwright
```

### 5.3 Mode 2: GitHub Actions CI (`.github/workflows/ci.yml`)

GitHub Actions automatically runs `python infra/scripts/verify.py --no-gpu` on every push and pull request. When adding Week 4+ features, register `"w4"` in `MILESTONE_GATES` inside `infra/scripts/verify.py`—no changes to `.github/workflows/ci.yml` are ever required.

### 5.4 Mode 3: Interactive Browser User Testing (`infra/scripts/dev_stack.py`)

To open ports `:8080` (Gateway) and `:3000` (Next.js Web UI) and test interactively in your browser at `http://127.0.0.1:3000`:

```bash
# Option A: Default (With Live GPU — start vLLM on :8000 first)
bash infra/scripts/start_worker_wsl.sh
python infra/scripts/dev_stack.py

# Option B: Without GPU (--no-gpu — runs real FastAPI Gateway on :8080 with MockInferenceWorker)
python infra/scripts/dev_stack.py --no-gpu

# Option C: Docker Compose (With GPU vs. Without GPU)
docker compose -f infra/compose/docker-compose.yml --profile gpu up --build
NO_GPU=1 docker compose -f infra/compose/docker-compose.yml up --build
```
