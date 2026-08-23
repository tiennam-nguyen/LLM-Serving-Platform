# Contributing Guide

Welcome to the **Adaptive LLM Serving Platform** project repository.

This project is built and maintained by a two-student engineering team. This guide establishes the workflow, coding standards, and verification requirements to keep development smooth, maintainable, and aligned with project milestones.

---

## 1. Development Principles

1. **Modular Monolith First**: Keep clear internal module boundaries (`gateway`, `routing`, `workers`, `telemetry`, `control`, `benchmark`, `common`) within a single well-structured package. Avoid premature distributed microservices.
2. **Separation of Concerns**: The data-plane request path (client -> gateway -> router -> worker) must remain decoupled from control-plane and database queries.
3. **Reproducibility**: Benchmark configs and experiment results must always track git commit hash, environment, model, and generation parameters.
4. **Verified Code Only**: Never push unverified code or broken tests to `main`.

---

## 2. Local Setup

1. **Clone and Navigate**:
   ```bash
   git clone https://github.com/tiennam-nguyen/LLM-Serving-Platform.git
   cd LLM-Serving-Platform
   ```

2. **Create and Activate Virtual Environment**:
   - On Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - On Linux / macOS:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install Editable Package with Development Dependencies**:
   ```bash
   python -m pip install -e ".[dev]"
   ```

---

## 3. Quality & Verification Standards

Before committing or opening a pull request, run the following verification suite:

```bash
# 1. Run unit and smoke tests
python -m pytest

# 2. Run linter checks
python -m ruff check .

# 3. Run format check
python -m ruff format --check .
```

---

## 4. Branching & Commit Conventions

- **Default Branch**: `main` (always keep production-ready / stable).
- **Feature Branches**: Use descriptive branch names:
  - `feat/week1-vllm-setup`
  - `feat/gateway-skeleton`
  - `test/benchmark-runner`
  - `docs/architecture-update`
- **Conventional Commits**: Format commit messages consistently:
  - `feat: add worker adapter interface`
  - `fix: handle streaming disconnect in gateway`
  - `docs: update week 1 runbook`
  - `test: add routing policy unit tests`
  - `chore: update dependencies in pyproject.toml`
