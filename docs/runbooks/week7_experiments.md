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
