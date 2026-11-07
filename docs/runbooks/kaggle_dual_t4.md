# Kaggle dual-T4 validation

This experiment uses **two full independent model replicas on one Kaggle node**. It does not validate two-host LAN behavior, and its streaming checks are not a benchmark.

## Submit and retrieve

Prerequisites: authenticated Kaggle CLI with access to private Internet-enabled T4 x2 script runs. The checked-in metadata targets `tiennamnguyen11/llm-serving-dual-t4-validation`; to use a different account, change only the metadata owner to the authenticated account. Keep `is_private: true`. Do not copy credentials into the kernel or repository.

From the repository root:

```bash
kaggle kernels push -p infra/kaggle/dual_t4_validation --accelerator NvidiaTeslaT4 --timeout 3600
kaggle kernels status tiennamnguyen11/llm-serving-dual-t4-validation
kaggle kernels logs tiennamnguyen11/llm-serving-dual-t4-validation
kaggle kernels output tiennamnguyen11/llm-serving-dual-t4-validation -p .cache/kaggle-dual-t4/output
```

Poll status at a bounded cadence; use the versioned slug for logs when available. Submit once and inspect before changing anything. Never push the Git experiment branch unless the live run and local regressions pass.

## What the run checks

`infra/kaggle/dual_t4_validation/run.py` fetches the exact Git SHA in `REVISION` and checks that its checkout is clean. The harness itself is sent by Kaggle CLI, not cloned from the unpushed experiment branch. The evidence includes the submitted script SHA-256. Update `REVISION` deliberately when selecting a new revision.

The harness checks for exactly two visible T4s, installs `.[dev]` without the legacy GPU extra, lists stable binary vLLM releases via pip, and installs a binary wheel. If Kaggle's Python is outside the project's 3.11–3.12 range, it provisions an isolated uv-managed Python 3.12 environment. Version 1 observed Python 3.13.16 and pip rejected it; version 2 addresses only that observed blocker. `VLLM_REQUIREMENT` can pin a specific observed compatible stable version on replay. Record the actual resolved versions and CUDA/driver pair before interpreting the result. The repository's dependency definition stays unchanged unless a demonstrated incompatibility requires a separate fix. See [vLLM CUDA installation requirements](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/index.html?device=cuda).

Version 2's pip resolver selected `vllm 0.33.0` and reported `transformers>=5.16.1,<5.21.0`, which cannot coexist with the original direct `transformers<4.49.0` requirement. The experiment branch removes that direct cap so vLLM declares its own compatibility range. `uv lock` updated the metadata without changing any of the 147 locked package identities. This local dependency correction was CPU-verified; the Kaggle kernel still checks out the original `db60212` and installs its serving wheel separately.

Each process sees one GPU through `CUDA_VISIBLE_DEVICES`, uses TP=1, and serves `Qwen/Qwen2.5-3B-Instruct` at FP16, memory utilization 0.85, context 4096. Worker ports are 8000 and 8001; the gateway is on 8080 with explicit worker IDs and `round_robin`. Readiness, installations, and process termination have time bounds.

The repository's `verify_week4_e2e.py --gateway-url http://127.0.0.1:8080 --model Qwen/Qwen2.5-3B-Instruct --require-multi-worker --save-evidence` establishes the live routing/trace baseline. The harness checks four additional requests in each of three phases: both replicas healthy; worker 0 process group stopped and registry exclusion observed; worker 0 restarted and both healthy again. Per-phase GPU process listings and process-group ownership must agree with the physical GPU UUIDs. Each request requires nonempty streamed content and matching completed traces with all six milestones.

## Evidence and diagnosis

The `/kaggle/working` output contains environment and validation JSON, the Week 4 live verifier JSON, three GPU process listings, and separate worker/gateway logs as the run reaches those phases. On failure, retain the diagnostic state and inspect the relevant logs. Absence of an artifact means its phase was not reached. A successful Kaggle scheduling status alone is not acceptance: check `passed`, the GPU mapping, all three routing sequences, and request correlations.

Use local CPU checks before and after changes:

```bash
python infra/scripts/verify.py --target lint --no-gpu
python infra/scripts/verify.py --target unit --no-gpu
python infra/scripts/verify.py --target integration --no-gpu
python infra/scripts/verify.py --target w4 --no-gpu
```

Keep downloaded logs under ignored `.cache/`. Only compact audited results belong in `experiments/raw/`; a failed attempt is not a passing evidence commit.

## Accepted run: 2026-11-07

[Dữ kiện] The private Kaggle kernel `tiennamnguyen11/llm-serving-dual-t4-validation`, version 2, completed against `db60212fadeaf71f8c0092a65dffd6c5629fa1d0`. The downloaded run recorded the exact harness SHA-256 `326343bcad3f08dcacb0738f19baa61264a6943eb9878b1803635d7dd3055ffa`. The submitted harness is unchanged from that run. The command above uses a fresh dependency resolution; to replay the observed wheel explicitly, set `VLLM_REQUIREMENT = "vllm==0.33.0"` before submission and compare the recorded resolved environment.

Both GPUs reported Tesla T4, 15,360 MiB each, driver `580.178.04`. Serving used Python `3.12.16`, PyTorch `2.14.0+cu130`, CUDA `13.0`, vLLM `0.33.0`, Transformers `5.20.2`, FastAPI `0.136.3` and httpx `0.28.1`. The primary 3B model ran with the original FP16/0.85/4096 configuration and TP=1; no model or memory fallback was needed.

| Phase | Registry (`worker-vllm-0`, `worker-vllm-1`) | GPU engine PIDs (GPU 0, GPU 1) | Four request assignments |
| --- | --- | --- | --- |
| Baseline | HEALTHY, HEALTHY | 291, 516 | 0, 1, 0, 1 |
| Worker 0 stopped | UNHEALTHY, HEALTHY | none, 516 | 1, 1, 1, 1 |
| Worker 0 restarted | HEALTHY, HEALTHY | 709, 516 | 0, 1, 0, 1 |

The built-in live verifier passed a separate four-request strict cycle. All twelve additional phase requests returned HTTP 200, nonempty SSE, matching request and worker IDs, completed traces, and six milestones. A local artifact audit matched all twelve trace IDs to HTTP 200 entries in the downloaded gateway access log and independently checked the GPU UUID/PID snapshots and source hash. Compact evidence is retained in `experiments/raw/kaggle_dual_t4/`; raw worker, gateway, and kernel logs are local and ignored.

[RAN] The four CPU targets above passed on the original branch base and on the final changes: Ruff lint, 39 unit tests, 8 integration tests, and the W4 mock gate. Verification used the isolated local CPU Python environment; no local CUDA dependencies were installed.

[INFERRED] The live trace and process evidence supports independent same-node replica routing and exclusion/re-entry after a real process stop/restart. [UNCHECKED] Two-host LAN behavior, network partitions, in-flight retry guarantees, and performance under load were not tested.
