# Week 4 Member B handoff

Member B control-plane slice implemented; Week 4 integration/exit gate pending Member A routing/gateway merge.

## Runtime boundary

`control/registry.py:WorkerRegistry` owns in-memory metadata and runtime references.
`register(worker, endpoint, model)` captures `worker.worker_id` once. Worker IDs must remain stable after registration. IDs, endpoint and model must be nonempty strings. Duplicate IDs raise `DuplicateWorkerError`; unknown lookups/health updates raise `UnknownWorkerError`.

`get_record(id)` and `list_workers()` return frozen `WorkerRecord` snapshots (the latter in a new list sorted by worker ID). Old records remain unchanged after updates. Records contain only `worker_id`, `endpoint`, `model`, `status`, `last_checked_at`; no runtime object is serialized. `healthy_workers()` returns a new list of the **actual** registered `InferenceWorker` instances with cached HEALTHY status, sorted by ID. This is the future routing candidate source: it performs no probes or network/database calls.

`record_health(id, healthy, checked_at=...)` accepts a boolean and an aware timestamp, normalized to UTC; absent timestamps use current UTC. The latest call wins, including when the wall clock moves backwards. Registration starts at STARTING/null; completed successful probes produce HEALTHY, false/exception/timeout produce UNHEALTHY. Recovery is the next successful probe. No expiry/staleness policy is implied.

`control/health.py:HealthMonitor(registry, timeout_s=5.0, clock=...)` exposes async `check_once()`. It snapshots `probe_targets()`, probes workers concurrently, and publishes each result when that probe finishes. Calls on one monitor are serialized. Ordinary probe exceptions are logged without exception contents and mapped to UNHEALTHY. Cycle cancellation propagates without fabricating a failed probe. Async workers must yield to the event loop and honor cancellation for the per-worker `asyncio.wait_for` timeout to work; Python cannot preempt a blocking/noncooperative implementation. Use one monitor per registry. No periodic loop or lifecycle task is installed.

## Read-only HTTP contract

Member A can mount the independently tested router with:

```python
from llm_serving_platform.control.api import create_control_router
from llm_serving_platform.control.registry import WorkerRegistry

registry = WorkerRegistry()
app.include_router(create_control_router(registry))
```

- `GET /control/v1/workers` → 200, a JSON **array** sorted by worker ID; empty registry → `[]`.
- `GET /control/v1/workers/{worker_id}` → 200, one record; unknown ID → 404 `{"detail":"Worker not found"}`.
- No mutation endpoints. Metadata endpoints never trigger probes.

Example record:

```json
{"worker_id":"mock-a","endpoint":"mock://a","model":"test-model","status":"STARTING","last_checked_at":null}
```

Status strings are exactly STARTING/HEALTHY/UNHEALTHY. Completed checks use aware UTC ISO-8601 timestamps (e.g. `2026-11-02T00:00:00Z`). Endpoints are display metadata, not browser fetch targets or links. IDs should be URL-safe single path segments for the detail route. Registry storage is process-local: multi-process sharing, removal and dynamic runtime replacement are outside this contract.

## Frontend and integration ownership

`/workers` uses existing plate/recess styles, a typed validated client, manual Refresh, a ten-second request deadline, and independent loading/empty/error/status states. No polling or editing. Playground links to Workers; Workers links back. Fetch failure cannot affect the chat session.

`NEXT_PUBLIC_CONTROL_API_URL` is an **origin/base URL**, not the full workers path. If unset/blank, the client falls back to `NEXT_PUBLIC_GATEWAY_URL`, then the gateway default `http://127.0.0.1:8080`. It appends `/control/v1/workers`. These public settings are build-time values. If using a separate origin, Member A must provide appropriate CORS. If gateway base is the optional `/api/gateway` relay, configure the control origin explicitly: no control relay route is added by this slice.

Until Member A mounts the router, an ordinary Week 3 gateway returns 404 and the Workers page honestly shows an error. Member A owns registration bootstrap, health scheduling/lifespan and shutdown, CORS, router mounting, and Round-Robin consuming cached healthy candidates. None is wired here. Gateway handling, routing implementation, CI workflow and `MILESTONE_GATES` remain unchanged.

## Verification

New contracts are in `tests/unit/test_control.py` and `apps/web/tests/workers.test.tsx`, discovered by the existing pytest/Vitest configuration and CPU CI pipeline. They use mocks/events and in-process ASGI; no GPU or live server is required. No W4 cumulative gate is registered. See the verification results appended after execution below.

### Observed local results (2026-11-02)

Base revision: `d5e42053095fc1b505f0b38d5406c3897e2e41f7`; branch `feat/week4-registry-health-ui`. Verification covers this slice's source/tests; only documentation text changed after the full run.

- [RAN] Existing Ubuntu Python3.12.3 environment: `python -m pytest tests/unit/test_control.py -q` → 9 passed; `python infra/scripts/verify.py --target lint --no-gpu` → Ruff passed; `python infra/scripts/verify.py --target unit --no-gpu` → 32 passed. Initial new-test line-length diagnostics were fixed by formatting.
- [RAN] Windows Node24.13.0/npm11.6.2: `npm test -- tests/workers.test.tsx` → 4 passed; `npm run lint` and `npm run typecheck` → exit0. The sandbox initially blocked Vitest startup with spawnEPERM; rerunning with host permission succeeded.
- [RAN] Full `python infra/scripts/verify.py --no-gpu` → ALL REQUESTED VERIFICATION CHECKS PASSED: Ruff; 32 unit and7 integration tests; W1–3 mock gates; frontend lint/typecheck; 30 Vitest tests; production build including `/workers`; 3 existing Playwright browser tests. Used ignored Windows `.cache/week4-cpu` with Python3.11, FastAPI0.162.0, Pydantic2.14.2, pytest9.2.0, pytest-asyncio1.4.1.post1 and Ruff0.17.3. Checked `find_spec`: both torch andvllm absent. This environment installed only the existing CI CPU dependencies; manifests/lockfiles were not changed. An early invocation before installation completed failed on missinghttpx; the completed-environment rerun passed. pytest-asyncio emitted its existing unset fixture-loop-scope deprecation warning.
- [READ] CI already discovers these test paths and runs the same `--no-gpu` entrypoint. Workflow and milestone registrations unchanged.
- [UNKNOWN] The installed local Node is below the frontend's existing declared engine range; passing local observations do not certify that range. CI's configured supported Node was not run locally. Live GPU, LAN, Round-Robin and integrated Week4 acceptance were intentionally not run.

No application process is launched by this slice's tests. The existing full pipeline briefly starts its Week3 CPU fixture servers for browser smoke checks; this does not constitute Week4 gateway/registry integration.
