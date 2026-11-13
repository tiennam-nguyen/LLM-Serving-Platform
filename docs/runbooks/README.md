# Operational Runbooks

This directory contains operational step-by-step procedures, deployment instructions, and troubleshooting guides for running the **Adaptive LLM Serving Platform** in local lab and multi-GPU environments.

---

## Runbook Catalog

| Runbook | Status | Purpose |
| :--- | :--- | :--- |
| [week5_lan_poc.md](week5_lan_poc.md) | Lab preparation; physical run pending | Two Linux/RTX 2080 hosts, LAN health, Round-Robin, manual failure and recovery evidence. |
| `01-week1-single-gpu-vllm.md` | *Planned (Week 1)* | Setting up single-GPU vLLM baseline, driver verification, and initial streaming test. |
| `02-gateway-deployment.md` | *Planned* | Launching the unified gateway and registering local/remote workers. |
| `03-multi-gpu-cluster-setup.md` | *Planned* | Bringing up multi-worker GPU nodes across lab workstations. |
| `04-benchmark-execution.md` | *Planned* | Running reproducible trace-driven workload experiments. |
| `99-troubleshooting.md` | *Planned* | Common issues: CUDA OOM, NCCL timeout, worker disconnections. |
