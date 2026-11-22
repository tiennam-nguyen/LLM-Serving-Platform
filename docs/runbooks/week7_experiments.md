# Week 7 Observability & Experiment Runner Runbook

## 1. Run an Experiment via CLI

Start a Gateway (mock or live GPU), then run any `schema_version: 1` manifest:

```bash
python -m llm_serving_platform.benchmark run \
  --manifest experiments/manifests/w7_smoke_least_loaded.json
```

Optional overrides:

```bash
python -m llm_serving_platform.benchmark run \
  --manifest experiments/manifests/w7_smoke_least_loaded.json \
  --gateway-url http://127.0.0.1:8080 \
  --output-dir experiments/runs
```

Exit codes:
- `0`: experiment completed (`run_status == "completed"`), all canonical artifacts written.
- `1`: fatal runtime/infrastructure/contract failure (`run_status == "failed"`), partial evidence preserved.
- `2`: invalid manifest, invalid workload JSONL, or colliding explicit `--run-id`.

Inspect live Gateway observability endpoints directly:

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/control/v1/workers
curl -s http://127.0.0.1:8080/metrics
```

---

## 2. CPU/Mock Verification Gates

Run Person A's focused observability contract gate:

```bash
python infra/scripts/verify.py --target w7-observability --no-gpu
```

Run the full Week 7 CPU/mock end-to-end experiment pipeline gate (manifest validation, live loopback mock multi-worker Gateway, warm-up exclusion, measured execution, `traces.jsonl`/`metrics.prom`/`summary.json` validation, prompt privacy check, partial-failure preservation, and Week 6 regression tests):

```bash
python infra/scripts/verify.py --target w7 --no-gpu
```

Run all implemented CPU/mock verification stages:

```bash
python infra/scripts/verify.py --no-gpu
```

Hosted GitHub Actions CI runs `lint`, `unit`, `integration`, implemented milestone gates (`w1`, `w2`, `w3`, `w4`, `w6`), the full Week 7 experiment gate (`w7`), and `web` (including Playwright) in CPU/mock mode. Week 5 two-machine LAN validation is intentionally kept separate from hosted CI.

---

## 3. Reproducible Dual-T4 Execution on Kaggle

The Week 7 Kaggle launcher (`infra/kaggle/week7_experiment_evidence/`) reuses the proven Week 6 dual-T4 worker/process-group harness (`infra/kaggle/dual_t4_validation/run.py`):

1. Verifies `nvidia-smi` and PyTorch/CUDA expose **exactly two physical NVIDIA Tesla T4 GPUs** (fails loudly if fewer than two T4s are allocated).
2. Starts two independent `vllm` (`0.33.0`, TP=1) replicas:
   - GPU 0 -> `worker-vllm-0` on `127.0.0.1:8000`
   - GPU 1 -> `worker-vllm-1` on `127.0.0.1:8001`
3. Proves GPU-to-worker process group bindings.
4. Runs `experiments/manifests/w7_t4_round_robin.json` and `experiments/manifests/w7_t4_least_loaded.json` against separate fresh Gateway instances on `127.0.0.1:8080`.
5. Validates all generated artifacts and keeps service logs separate from experiment evidence.

### Packaging & Submitting from a Clean Commit

```bash
python infra/kaggle/week7_experiment_evidence/package_kernel.py \
  --owner tiennamnguyen11 \
  --base <verified-public-ancestor> \
  --model Qwen/Qwen2.5-3B-Instruct \
  --vllm-version 0.33.0 \
  --output-dir .cache/kaggle-week7/kernel

kaggle kernels push \
  -p .cache/kaggle-week7/kernel \
  --accelerator NvidiaTeslaT4 \
  --timeout 3600

kaggle kernels status tiennamnguyen11/llm-serving-week-7-experiment-evidence
kaggle kernels output tiennamnguyen11/llm-serving-week-7-experiment-evidence \
  -p .cache/kaggle-week7/output
```

> [!NOTE]
> Live Week 7 dual-T4 runs validate the end-to-end observability and experiment artifact pipeline on real GPUs. They are **not** a policy performance comparison or benchmark winner claim.

---

## 4. Accepted Dual-T4 Run, 2026-11-22

- **Kaggle kernel**: [`tiennamnguyen11/llm-serving-week-7-experiment-evidence`](https://www.kaggle.com/code/tiennamnguyen11/llm-serving-week-7-experiment-evidence), version 1
- **Tested clean Git SHA**: `df03a276cddf94703a3482ed9a85422ba6f10ec4` (`git_dirty: false`)
- **Bundle SHA-256**: `990d5af74eef0df74b1e18204ed00854c83bd5f519f80042ea77af16b7b4fc91`
- **Submitted script SHA-256**: `0c12d741b606d1d544a588b566d86b9ad65bb573643f7d04fce47e7d8a35ed10`
- **Hardware**: 2 × NVIDIA Tesla T4 (`15,360 MiB` each, driver `580.178.04`), PyTorch `2.14.0+cu130`, CUDA `13.0`, vLLM `0.33.0`, Python `3.12.16`
- **Model & Topology**: `Qwen/Qwen2.5-3B-Instruct`, tensor parallel size 1 per replica:
  - GPU 0 (`GPU-2e551fb9-9fd2-4a9c-9ce2-3474cd652468`) -> `worker-vllm-0` (`http://127.0.0.1:8000`)
  - GPU 1 (`GPU-526f9bad-7541-6122-fdba-3485f7c9ce87`) -> `worker-vllm-1` (`http://127.0.0.1:8001`)
- **Round-Robin run (`20261122T045731Z-w7-t4-round-robin-df03a27-14295e`)**: 20/20 completed (2 warm-up excluded; 22 total in `metrics.prom`), worker distribution `worker-vllm-0: 10`, `worker-vllm-1: 10`, final active requests `0`/`0`.
- **Least-Loaded run (`20261122T045755Z-w7-t4-least-loaded-df03a27-4ed5ce`)**: 20/20 completed (2 warm-up excluded; 22 total in `metrics.prom`), worker distribution `worker-vllm-0: 9`, `worker-vllm-1: 11`, final active requests `0`/`0`.

All 12 downloaded live artifact files are preserved byte-for-byte under [`experiments/raw/week7/live/`](../../experiments/raw/week7/live/) and passed independent local artifact validation (`summary.json` recomputed from `traces.jsonl`, Prometheus parsing, zero active requests, zero prompt/secret leakage).
