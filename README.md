# Adaptive LLM Serving Platform for Observable Multi-GPU Inference

## 1. Overview

The **Adaptive LLM Serving Platform** is an observable, reproducible, multi-worker LLM inference serving platform designed to manage, trace, and route LLM workloads across GPU nodes.

The system follows a **modular monolith** architecture with strict separation of concerns (`SOLID`) between the **data-plane HTTP/SSE gateway**, **control-plane worker registry & concurrent health monitor (`/control/v1/workers`)**, **pluggable routing policies (`RoundRobinRouter` & `SingleWorkerRouter`)**, **runtime-agnostic worker adapters (`create_worker` & `populate_registry`)**, **6-milestone per-request telemetry**, **cumulative Open-Closed verification gates (`infra/scripts/verify.py`)**, and the **Next.js operator console (`apps/web` Playground & `/workers` Topology view)**.

---

## 2. Current Status: Week 4 Completed (Multi-Worker Registry, Health Monitor & Round-Robin)

> [!NOTE]
> **Status**: Week 4 Complete — Multi-Worker Control Plane, Concurrent Health Monitor, Round-Robin Routing & `/workers` Topology UI (`Browser → Gateway → RoundRobinRouter → Worker Pool [GPU / No-GPU]`).
>
> - **Week 1 (Single-GPU vLLM Baseline)**: Completed (`Qwen/Qwen2.5-0.5B-Instruct` on CUDA `float16`, launch scripts, latency measurement).
> - **Week 2 (Data-Plane Inference Gateway)**: Completed (OpenAI-compatible `POST /v1/chat/completions` SSE proxy, `InferenceWorker` / `VLLMWorkerAdapter`, `X-Request-ID`, structured JSON logging, timeout/error propagation).
> - **Week 3 (Minimal End-to-End Web Flow, Request Trace & Unified Verification)**: Completed (Next.js Instrumentation Deck UI, multi-turn streaming, abort/retry, `X-Worker-ID`, 6-milestone `RequestTrace` via `GET /v1/traces/{request_id}`, `create_worker` factory, unified `verify.py` runner, and `dev_stack.py` interactive browser launcher).
> - **Week 4 (Worker Registry, Health Monitor, Round-Robin & Workers Topology UI)**: Completed (`WorkerRegistry`, `HealthMonitor` with startup + periodic background checks, `GET /control/v1/workers` & `GET /control/v1/workers/{id}`, `RoundRobinRouter` with automatic unhealthy-worker exclusion & HTTP 503 empty-pool handling, `/workers` Web UI, and cumulative gate `w4`).
> - **Next Milestone**: Week 5 — Two-machine LAN PoC: remote endpoints, health over LAN, Round-Robin across hosts, failure/restart recovery.

Week 5 lab preparation: follow the [two-host Linux/RTX 2080 runbook](docs/runbooks/week5_lan_poc.md)
using `.env.week5.example`, `start_worker_lab.sh`, `start_gateway_lab.sh` and
`verify_week5_lan.py`. Physical LAN acceptance requires the three lab phases;
CPU/mock verification alone does not complete Week 5.

---

## 3. Semester Engineering Progression

```text
[x] Week 1: 1 GPU / 1 worker (vLLM baseline)
    ↓
[x] Week 2: Unified data-plane gateway + VLLMWorkerAdapter
    ↓
[x] Week 3: Web E2E operator/chat UI + 6-milestone RequestTrace + Unified Verify/Stack CLI
    ↓
[x] Week 4: Worker registry + health monitor + Round-Robin routing + /workers UI
    ↓
[ ] Week 5: Two-machine LAN PoC (remote endpoints + failure/restart recovery)
    ↓
[ ] Week 6: Primary multi-worker testbed + Least-Loaded routing
    ↓
[ ] Week 7: Observability + experiment artifacts
    ↓
[ ] Week 8: Controlled benchmark suite
    ↓
[ ] Week 9: P0 consolidation + optional Cache-Only (BUFFER + feature freeze)
    ↓
[ ] Week 10: Final experiment campaign
    ↓
[ ] Week 11: Hardening + release candidate + report integration
    ↓
[ ] Week 12: Final release + defense-ready package
```

---

## 4. Repository Layout

```text
.
├── src/
│   └── llm_serving_platform/       # Root Python package (modular monolith)
│       ├── __init__.py
│       ├── __main__.py             # Bootstrap verification CLI entrypoint
│       ├── gateway/                # FastAPI HTTP/SSE entrypoint (main, schemas, streaming)
│       ├── routing/                # RoutingPolicy ABC, RoundRobinRouter & SingleWorkerRouter
│       ├── workers/                # InferenceWorker ABC, VLLM/Mock adapters, pool factory
│       ├── telemetry/              # 6-milestone RequestTrace & InMemoryTraceStore
│       ├── control/                # WorkerRegistry, HealthMonitor & /control/v1/workers API
│       ├── benchmark/              # Workload replay & benchmark orchestration boundary
│       └── common/                 # 12-Factor GatewayConfig (.env + multi-worker loader) & logging
│
├── apps/
│   └── web/                        # Next.js / React / TypeScript Playground (/) & Workers (/workers)
│
├── infra/
│   ├── compose/                    # Hardened Dockerfiles & Docker Compose (GPU profile & NO_GPU=1)
│   ├── monitoring/                 # Monitoring and metrics configuration
│   └── scripts/                    # Unified verify.py (w1-w4), dev_stack.py, and milestone scripts
│
├── tests/
│   ├── conftest.py                 # Pytest --no-gpu and --prompt fixtures
│   ├── mocks.py                    # Re-exports canonical MockInferenceWorker
│   ├── unit/                       # Unit tests (control, routing, validation, CORS, traces, package)
│   └── integration/                # Integration & dual-mode E2E tests (Week 3 & Week 4)
│
├── docs/
│   ├── architecture.md             # Week 4 Architecture Snapshot & SOLID boundaries
│   ├── report_outline_week3.md     # Week 3 Report Outline & milestone evidence summary
│   ├── report_outline_week4.md     # Week 4 Report Outline & milestone evidence summary
│   ├── week4-member-b-contract.md  # Week 4 Control-Plane & UI boundary contract
│   └── runbooks/setup_wsl.md       # Operational runbook (WSL2 + vLLM GPU setup)
│
├── .github/workflows/ci.yml        # GitHub Actions CI (supports No-GPU default & GPU dispatch)
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

`infra/scripts/verify.py` is the **single source of truth** for all testing (Lint, Unit, Integration, Cumulative Weekly Gates `w1` → `w2` → `w3` → `w4`, and Frontend/Playwright).
- **Default**: Runs in **Live GPU mode** (`http://127.0.0.1:8080` → vLLM `:8000`).
- **`--no-gpu`**: Runs in **No-GPU mode** using a multi-worker `MockInferenceWorker` pool.

```bash
# A. Full End-to-End Verification (All stages + Week 1-4 Gates + Web E2E)
python infra/scripts/verify.py                # Default: Live GPU
python infra/scripts/verify.py --no-gpu       # No-GPU (CPU / CI mode)

# B. Partial / Component Verification (both GPU default and --no-gpu supported)
python infra/scripts/verify.py --target lint --no-gpu
python infra/scripts/verify.py --target unit --no-gpu
python infra/scripts/verify.py --target integration --no-gpu
python infra/scripts/verify.py --target w1 --no-gpu      # Week 1 Gate only (omit --no-gpu for GPU)
python infra/scripts/verify.py --target w2 --no-gpu      # Week 2 Gate only
python infra/scripts/verify.py --target w3 --no-gpu      # Week 3 Gate only
python infra/scripts/verify.py --target w4 --no-gpu      # Week 4 Gate only (Registry + Round-Robin + Failover)
python infra/scripts/verify.py --target gates --no-gpu   # All cumulative weekly gates (W1 -> W4)
python infra/scripts/verify.py --target web --no-gpu     # Frontend lint, typecheck, vitest, build, Playwright
```

### 5.3 Mode 2: GitHub Actions CI (`.github/workflows/ci.yml`)

GitHub-hosted CI is intentionally **CPU/mock-only**. It exposes five named checks in the Actions UI—Python lint/static, backend unit, backend integration, cumulative W1→W4 gates, and frontend/browser—and publishes the same results in the GitHub job summary. This keeps CI deterministic and does not pretend a hosted runner has access to the lab/Kaggle GPU topology.

Real GPU validation is run manually on actual hardware. Week 4 does **not** require two physical GPUs; its full failure/recovery gate is verified deterministically with mock workers. When 2+ live workers are available, use:

```bash
python infra/scripts/verify_week4_e2e.py --gateway-url http://127.0.0.1:8080 --require-multi-worker --save-evidence
```

That command strictly checks the live healthy-worker snapshot, cyclic Round-Robin placement, CORS, request IDs, and trace correlation. Active worker stop/restart and recovery across hosts are deliberately deferred to the **Week 5 two-machine LAN PoC**.

### 5.4 Mode 3: Interactive Browser User Testing (`infra/scripts/dev_stack.py`)

To open ports `:8080` (Gateway) and `:3000` (Next.js Web UI) and test interactively in your browser at `http://127.0.0.1:3000` (Playground) and `http://127.0.0.1:3000/workers` (Workers Topology):

```bash
# Option A: Default (With Live GPU — start vLLM on :8000 first, or pass --worker-endpoints for multiple GPUs)
bash infra/scripts/start_worker_wsl.sh
python infra/scripts/dev_stack.py

# Option B: Without GPU (--no-gpu — runs real FastAPI Gateway on :8080 with 2x MockInferenceWorker + Round-Robin)
python infra/scripts/dev_stack.py --no-gpu

# Option C: Docker Compose (With GPU vs. Without GPU)
docker compose -f infra/compose/docker-compose.yml --profile gpu up --build
NO_GPU=1 docker compose -f infra/compose/docker-compose.yml up --build
```
