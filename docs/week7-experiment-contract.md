# Week 7: Observability & Experiment Artifact Contract

This document defines the Week 7 observability architecture, telemetry privacy boundary, versioned experiment manifest schema, workload format, and canonical run artifact contract.

---

## 1. Architecture Overview

The experiment pipeline treats the Gateway strictly as a black-box HTTP service. The `llm_serving_platform.benchmark` package never imports or instantiates Gateway internals (`RoundRobinRouter`, `LeastLoadedRouter`, `WorkerRegistry`, or `InMemoryTraceStore`).

```text
machine-readable manifest (.json, schema_version=1)
        │
        ▼
ExperimentConfig (strict Pydantic validation, extra="forbid")
        │
        ▼
ExperimentRunner (HTTP-only boundary)
        ├── GET  /health                   (preflight reachability & health)
        ├── GET  /control/v1/workers       (preflight topology & model check)
        ├── POST /v1/chat/completions      (warm-up requests, excluded from measured data)
        ├── POST /v1/chat/completions      (measured concurrent SSE requests, stream=true)
        ├── GET  /v1/traces/{request_id}   (bounded polling for terminal request traces)
        └── GET  /metrics                  (final Prometheus snapshot)
        │
        ▼
experiments/runs/<run_id>/
        ├── manifest.json                  (normalized resolved manifest)
        ├── environment.json               (safe git/runtime/hardware/worker metadata)
        ├── traces.jsonl                   (authoritative per-request terminal traces)
        ├── metrics.prom                   (raw final Prometheus exposition snapshot)
        └── summary.json                   (strictly recomputed from traces.jsonl)
```

---

## 2. Observability Signal Responsibilities

| Signal | Endpoint / Sink | Responsibility | Cardinality / Privacy Rules |
| --- | --- | --- | --- |
| **Traces** | `GET /v1/traces/{request_id}` | Authoritative per-request lifecycle milestones (`arrival`, `routing_start`, `routing_end`, `worker_start`, `first_token`, `last_token`), derived `metrics_ms`, routing candidate snapshot, and controlled error classification. | Bounded FIFO in-memory store (`max_size=500`). Never stores raw prompt text or arbitrary upstream exception text. |
| **Metrics** | `GET /metrics` | Per-app Prometheus `CollectorRegistry` cumulative counters, latency histograms, and cached worker gauges. | Low-cardinality labels only (`status`, `policy`, `worker_id`). Never labels by `request_id` or prompt text. Scraping is side-effect free and never triggers network health probes. |
| **Structured Logs** | `llm_serving_platform.gateway` JSON logger | Correlated lifecycle events keyed by `request_id`, `worker_id`, `model`, `status`, and safe error classifications. | Never logs full request payloads, prompt contents, or raw upstream exception messages. |

### Prometheus Metric Names

Exposed at `GET /metrics` (`text/plain; version=0.0.4; charset=utf-8`):

- `llm_gateway_requests_total{status}` — terminal admitted inference requests (`completed`, `error`, `cancelled`).
- `llm_gateway_request_e2e_seconds` — histogram of arrival-to-terminal request latency.
- `llm_gateway_ttft_seconds` — histogram of arrival-to-first-content-chunk latency (observed only when content arrived).
- `llm_router_decision_seconds{policy}` — histogram of worker selection/reservation duration (`single`, `round_robin`, `least_loaded`).
- `llm_router_requests_total{policy,worker_id}` — counter of successful worker selections.
- `llm_worker_active_requests{worker_id}` — gauge of current Gateway-owned active requests (must return to `0` when idle).
- `llm_worker_health{worker_id}` — gauge of cached worker health (`1` = `HEALTHY`, `0` otherwise).

### Privacy & Controlled Error Vocabulary

To prevent raw, partial, or transformed prompt echoes in upstream worker errors from defeating the storage boundary, the Gateway persists only a small controlled vocabulary in `RequestTrace` (`error`, `error_type`, `error_code`) and structured logs:

| `error` | `error_type` | Typical `error_code` | Meaning |
| --- | --- | --- | --- |
| `no_healthy_worker` | `WorkerUnavailableError` | `503` | No `HEALTHY` worker available at routing time |
| `routing_error` | Exception class name | `500` | Unexpected exception at the routing boundary |
| `worker_unavailable` | `WorkerUnavailableError` | Upstream HTTP / `503` | Worker unreachable or returned non-200 |
| `worker_timeout` | `WorkerTimeoutError` | `504` | Worker exceeded configured timeout |
| `worker_error` | `WorkerException` subclass | Upstream HTTP / `500` | Other worker-level exception |
| `worker_sse_error` | `WorkerSSEError` | SSE integer code or `null` | Worker emitted an SSE `{"error": ...}` chunk |
| `client_disconnected` | `ClientDisconnected` | `null` | Client disconnected or ASGI send failed |
| `response_setup_error` | Exception class name | `500` | Error constructing streaming response after admission |
| `internal_error` | Exception class name | `500` | Unexpected internal error during stream execution |

Client-facing SSE error messages remain informative where appropriate, while observability traces, logs, metrics, and experiment artifacts exclude prompt text and secrets.

---

## 3. Experiment Manifest Contract (`schema_version: 1`)

Manifests are JSON objects validated strictly with Pydantic (`extra="forbid"` — unknown keys are rejected for reproducibility).

```json
{
  "schema_version": 1,
  "name": "w7-smoke-least-loaded",
  "model": {
    "id": "Qwen/Qwen2.5-3B-Instruct"
  },
  "topology": {
    "worker_count": 2
  },
  "policy": {
    "name": "least_loaded"
  },
  "workload": {
    "path": "benchmarks/workloads/w7_smoke.jsonl",
    "request_count": 20
  },
  "seed": 42,
  "warmup": {
    "request_count": 2
  },
  "generation": {
    "temperature": 0,
    "max_tokens": 64
  },
  "runner": {
    "concurrency": 2,
    "gateway_url": "http://127.0.0.1:8080",
    "request_timeout_s": 120
  }
}
```

Validation rules:
- `schema_version`: integer `1` (unsupported versions fail fast).
- `policy.name`: one of `"single"`, `"round_robin"`, `"least_loaded"`.
- `topology.worker_count`, `workload.request_count`, `runner.concurrency`, `generation.max_tokens`: integers `>= 1` (booleans rejected).
- `warmup.request_count`: integer `>= 0`.
- `workload.path`: resolved safely against repository/base root; relative path-traversal escapes (`..`) are rejected.
- `runner.gateway_url`: `http://` or `https://` URL with host, no embedded credentials, normalized without trailing slash.

---

## 4. Workload JSONL Format

Each non-empty line is a JSON object with a non-empty `messages` array of `{"role": "...", "content": "..."}` entries (optional `"id"` string allowed; extra keys rejected):

```json
{"messages":[{"role":"user","content":"Explain what TTFT means in LLM serving."}]}
{"messages":[{"role":"user","content":"Give one sentence about request routing."}]}
```

- Malformed JSON or empty `role`/`content` fails fast with a 1-based line number.
- The workload must contain at least `workload.request_count` valid entries.
- Measured entries are ordered deterministically using `random.Random(seed)`.
- Input workload files may contain prompts; generated run artifacts (`manifest.json`, `environment.json`, `traces.jsonl`, `metrics.prom`, `summary.json`) never persist prompt text.

---

## 5. Canonical Artifact Directory Contract

Default output root: `experiments/runs/<run_id>/` (ignored by Git).
Each `run_id` is collision-resistant and human-readable: `<YYYYMMDDTHHMMSSZ>-<slug>-<shortsha>-<6hex>`.

1. **`manifest.json`** — atomically written normalized configuration actually executed.
2. **`environment.json`** — atomically written safe reproducibility metadata (`run_id`, `started_at`, `ended_at`, `git_revision`, `git_dirty`, `branch`, `python_version`, `platform`, allowlisted installed package versions, optional `nvidia-smi` GPU inventory, `model_id`, `routing_policy`, `expected_worker_count`, sanitized `gateway_url`, `observed_workers`). Never dumps `os.environ`, `.env`, or tokens.
3. **`traces.jsonl`** — one compact JSON object per measured terminal Gateway trace (`completed`, `error`, or `cancelled`), appended and flushed as each request completes. Warm-up requests are excluded.
4. **`metrics.prom`** — raw final Prometheus snapshot from `GET /metrics`. Documented as a process-level cumulative snapshot; `summary.json` never derives per-request percentiles from cumulative process counters.
5. **`summary.json`** — generated strictly by re-reading `traces.jsonl` from disk. Uses deterministic linear interpolation (**NIST R-7 / Hyndman-Fan type 7**): for sorted finite samples $x_0 \le \dots \le x_{n-1}$, $k = \frac{p}{100}(n-1)$ and $q_p = x_{\lfloor k \rfloor} + (k - \lfloor k \rfloor)(x_{\lceil k \rceil} - x_{\lfloor k \rfloor})$. Empty sample sets return `null` (never `0`), and missing/non-applicable values are explicitly counted in `latency_sample_counts`.

### Failure & Partial-Run Semantics

- **Request-level inference failures** (SSE `{"error": ...}` under HTTP 200, or Gateway HTTP 503) are recorded as terminal `error` traces in `traces.jsonl`; the experiment run continues and completes (`run_status: "completed"`).
- **Fatal infrastructure/contract failures** (unreachable Gateway, preflight topology/model mismatch, warm-up failure, trace timeout, policy mismatch) abort further dispatching and mark `run_status: "failed"` with a controlled `failure.type` (never raw exception strings).
- The run directory is **never deleted** on failure: `manifest.json`, `environment.json`, partial `traces.jsonl`, `metrics.prom` (if obtainable), and `summary.json` are all preserved on disk.

---

## 6. Known Limitations (Week 7 Scope)

- Week 7 builds trustworthy measurement and artifact provenance infrastructure; it is **not** the final Week 8 benchmark campaign and intentionally omits policy winner claims, confidence intervals, goodput curves, or Cache-Aware routing.
- `InMemoryTraceStore` holds the most recent 500 traces in memory; the runner polls each request's trace immediately after stream completion so long runs do not lose traces to eviction.
- Week 5 two-machine LAN validation remains separate from hosted CPU/mock CI.
