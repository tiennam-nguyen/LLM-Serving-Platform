# Week 3 evidence ledger

Branch: feat/week3-web-playground. Initial worktree clean; main pull --ff-only reported already up to date.

READ: gateway/main.py:create_app exposes /health (healthy/up or 503 unhealthy/down), streaming chat, X-Request-ID. safe_event_generator reports errors in SSE data.error. schemas.py:ChatCompletionRequest requires model/messages; vllm_adapter.py:stream_chat forwards data lines. No CORS middleware is configured.

Scope: apps/web only; no backend edits. npm convention selected because no JS package configuration exists. Use configured model only, no default invented model. Client timing is monotonic, usage only from actual stream fields.

Completed: Next/React/TypeScript scaffold; parser and request lifecycle; semantic components using reference surface primitives; unit/component/browser tests; build and Python verification; runbook.

Origin experiment hypothesis: direct browser calls to a separately hosted gateway fail CORS; a same-origin streaming frontend relay preserves progressive delivery and cancellation. Falsification: browser direct call succeeds, or relay delivers first content only after upstream completion / fails to cancel upstream. Tests must distinguish a controlled fixture from live GPU verification.

## Verification results

Baseline commit: `00a139e51c793ac0c5063494eef9cbc7dde2b0e4`. Branch: `feat/week3-web-playground`. All tracked changes are inside `apps/web/`. Deterministic tests; no random workload seed and no benchmark claims. Package versions are registry-resolved and captured in `package-lock.json`.

RAN (Windows, Node 24.13.0, npm 11.6.2):

| Command | Observed result |
| --- | --- |
| `npm install next react react-dom react-markdown` | 107 packages added; audit found 0 vulnerabilities. |
| `npm install -D typescript @types/node @types/react @types/react-dom eslint eslint-config-next vitest jsdom @testing-library/react @testing-library/user-event @testing-library/jest-dom @playwright/test` | 409 packages added, 1 removed; audit found 0 vulnerabilities. |
| `npm run lint` | Exit 0; no lint diagnostics. |
| `npm run typecheck` | Route types generated; TypeScript exit 0. |
| `npm test` | 5 test files passed; 26 tests passed. |
| `npm run build` | Exit 0; static `/`, dynamic health/chat relay routes built. |
| `npm run test:browser` | 3 passed. Chromium; fixture model; Next at 3001, wire fixture at 8081. |
| `git diff --check` | Exit 0. |
| `python -m pytest` (default MSYS Python) | Failed: pytest missing. |
| `python -m ruff check .` / `python -m ruff format --check .` (default MSYS Python) | Failed: ruff missing. |
| `.\.venv\Scripts\python.exe -m pytest` | Initially collection failed: httpx missing. After installing repository-declared gateway/test dependencies: **23 passed**. |
| `.\.venv\Scripts\python.exe -m ruff check .` | All checks passed. |
| `.\.venv\Scripts\python.exe -m ruff format --check .` | **6 files would be reformatted, 40 already formatted**. Existing files listed below; unchanged by this branch. |

Existing formatting failures: `infra/scripts/measure_latency.py`, `infra/scripts/verify_week2_exit_gate.py`, `src/llm_serving_platform/workers/base.py`, `src/llm_serving_platform/workers/vllm_adapter.py`, `tests/integration/test_gateway_error_propagation.py`, `tests/mocks.py`.

Environment repair only (no backend dependency manifest changes): `.\.venv\Scripts\python.exe -m pip install fastapi httpx uvicorn pytest-asyncio`. This existing venv uses Python 3.11; the repository pyproject requests Python 3.12. The passing Python suite is evidence for this local test environment, not full supported-runtime/GPU certification.

## Origin and streaming evidence

RAN: actual gateway launched with `PYTHONPATH=src` and `.\.venv\Scripts\python.exe -m uvicorn llm_serving_platform.gateway.main:app --host 127.0.0.1 --port 8080`. Browser probe with `CORS_PROBE_URL=http://127.0.0.1:8080/health`, `npm run test:browser -- tests/browser/origin.spec.ts`, reported:

```text
TypeError: Failed to fetch
Access to fetch at 'http://127.0.0.1:8080/health' from origin 'http://127.0.0.1:3001' has been blocked by CORS policy: No 'Access-Control-Allow-Origin' header is present on the requested resource.
```

RAN: the same-origin relay browser test observed first content before the delayed terminal event, then verified the upstream fixture's disconnect count increased after browser Abort. Console evidence:

```text
RELAY: first content visible while upstream remains open
RELAY: browser abort closed upstream; partial response retained
```

RAN regression: the first relay rejected legitimate loopback requests. Diagnostic: `origin=http://127.0.0.1:3001 requestOrigin=http://localhost:3001 host=127.0.0.1:3001`. READ: installed `next/dist/server/web/spec-extension/adapters/next-request.js:fromNodeNextRequest` constructs the request URL from Next metadata. Comparing against actual Host plus scheme fixes the reproduced request while six relay tests retain rejection of foreign/malformed origins. Temporary diagnostic logging removed.

## UI verification boundary

RAN automated browser: page/ONLINE health, blank/whitespace prevention, multi-turn histories (1 then 3 messages at the wire), progressive text, Abort/upstream disconnect, partial retention, request-ID capture, inspector switching via menu and ledger, error/retry, markdown code bay, phone width without horizontal overflow. Fixture use is explicit; none of these are claimed as real vLLM inference.

RAN visual review: desktop and phone screenshots inspected (`test-results/desktop.png`, `test-results/phone.png`, ignored). In-app browser initially showed unconfigured/offline state and disabled Send. Later manual interaction against the fixture was blocked by the in-app browser's stale connection-error page/URL policy. Full manual interaction checklist is **NOT RUN**; the automated Chromium checklist above passed.

UNKNOWN / NOT RUN: real browser-to-vLLM/GPU multi-turn generation, GPU streaming/cancellation, server-provided usage on a live model, and hosted reverse-proxy behavior. No served model was supplied. No GPU runtime was configured for this task. Configure the exact model and live gateway with the README instructions to run those checks.

ASSUMED: a single configured model is sufficient for Week 3; model stays unset until supplied. Session-only memory, latest-branch continuation on retry, fixed-route same-origin relay are the chosen defaults.

READ: `src/lib/gateway/client.ts:streamChat` sources request IDs from response headers and forwards explicit SSE events; `src/lib/telemetry/request-record.ts:transition` records monotonic client timestamps and explicit usage; `src/components/telemetry/RequestInspector.tsx:RequestInspector` labels unavailable fields. No hardware, queue/prefill/decode, token estimates or aggregate history is fabricated.
