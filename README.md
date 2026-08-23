# Adaptive LLM Serving Platform for Observable Multi-GPU Inference

## 1. Overview

The **Adaptive LLM Serving Platform** is an observable, reproducible, multi-worker LLM inference serving platform designed to manage and route distributed LLM workloads across multiple GPU nodes.

The system emphasizes a **modular monolith** architecture, keeping a clean separation between data-plane inference routing and control-plane management, with serving runtimes sitting behind standardized worker adapters.

---

## 2. Current Status

> [!NOTE]
> **Status**: Initial Repository Bootstrap (Pre-feature development).
>
> This repository is currently at the initial bootstrap stage. No serving runtime, gateway proxy, GPU inference workload, or database is active yet. Week 1 implementation begins immediately following this bootstrap.

---

## 3. Semester Engineering Target

The semester goal is to deliver an observable, reproducible, multi-worker LLM serving platform capable of serving concurrent requests across heterogeneous/multi-GPU nodes with transparent metrics, reproducible benchmarks, and dynamic routing.

### Milestone Progression

```text
1 GPU / 1 worker (vLLM baseline)
    ↓
Unified gateway
    ↓
Web E2E (minimal operator/chat UI)
    ↓
Worker registry + health + Round-Robin routing
    ↓
LAN PoC (multi-node networking)
    ↓
2+ independent GPU replicas
    ↓
Least-Loaded routing policy
    ↓
Observability + reproducible benchmark runner
    ↓
Optional adaptive routing research (cache-aware / SLO-aware)
```

> [!IMPORTANT]
> The engineering baseline (multi-worker serving, observability, reproducible benchmarking) stands on its own. Advanced cache-aware or SLO-aware routing policies are exploratory and are **not on the critical path** for project success.

---

## 4. Repository Layout

```text
.
├── src/
│   └── llm_serving_platform/       # Root Python package (modular monolith)
│       ├── __init__.py
│       ├── __main__.py             # Bootstrap verification CLI entrypoint
│       ├── gateway/                # Future data-plane HTTP entrypoint
│       ├── control/                # Future control plane & operator management
│       ├── benchmark/              # Future workload replay & benchmark orchestration
│       ├── routing/                # Future RoutingPolicy abstractions & policies
│       ├── workers/                # Future runtime adapters (vLLM, mock)
│       ├── telemetry/              # Future metrics & state collection
│       └── common/                 # Shared cross-cutting primitives
│
├── apps/
│   └── web/                        # Future minimal Web UI walking skeleton
│
├── benchmarks/
│   ├── configs/                    # Workload generator configuration files
│   ├── traces/                     # Workload trace datasets
│   └── workloads/                  # Workload definitions and arrival patterns
│
├── experiments/
│   ├── manifests/                  # Experiment specification manifests
│   ├── raw/                        # Raw execution logs and latency records
│   ├── processed/                  # Aggregated metrics and processed tables
│   └── figures/                    # Generated charts and visualization figures
│
├── infra/
│   ├── compose/                    # Local multi-service compose specifications
│   ├── monitoring/                 # Monitoring and metrics configuration
│   └── scripts/                    # Lab environment and infrastructure setup scripts
│
├── scripts/                        # Developer utility scripts
│
├── tests/
│   ├── unit/                       # Unit tests and package structure verification
│   └── integration/                # Future multi-component integration tests
│
├── docs/
│   ├── adr/                        # Architecture Decision Records
│   ├── api/                        # API specifications & interface contracts
│   ├── experiments/                # Experiment methodology & reproducibility guides
│   ├── runbooks/                   # Operational runbooks & setup procedures
│   └── architecture.md             # System architecture & boundary design
│
├── .editorconfig                   # Editor and IDE formatting rules
├── .env.example                    # Template environment variables
├── .gitignore                      # Git exclusion rules
├── CONTRIBUTING.md                 # Contribution and workflow guidelines
├── SECURITY.md                     # Lab security and secret management policy
├── pyproject.toml                  # Python package configuration & dependencies
└── README.md                       # Project documentation entry point
```

---

## 5. Local Setup & Developer Commands

### Prerequisites

- Python 3.10+ (Python 3.11 recommended)
- Git

### Setup Virtual Environment

```bash
# 1. Create virtual environment
python -m venv .venv

# 2. Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 3. Install editable package with development tools
python -m pip install -e ".[dev]"
```

### Verification & Smoke Test

```bash
# Run package bootstrap entrypoint
python -m llm_serving_platform
# Expected output: LLM Serving Platform bootstrap OK
```

### Testing and Linting

```bash
# Run test suite
python -m pytest

# Run linter checks
python -m ruff check .

# Run code format check
python -m ruff format --check .
```

---

## 6. Next Milestone: Week 1

**Week 1 Target**: Environment setup and single-GPU vLLM serving baseline.
1. Lab environment verification (NVIDIA drivers, CUDA, PyTorch).
2. Direct standalone vLLM instance bring-up on 1 GPU.
3. First streamed request benchmark and metrics recording.
4. Record environment parameters and launch/cleanup runbook in `docs/runbooks/`.
