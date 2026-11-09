# Week 6 Person A -> Person B Handoff Contract

Person A core load-accounting and routing slice (`control/registry.py`, `routing/router.py`, `routing/__init__.py`, `common/config.py`, and unit tests in `tests/unit/test_week6_load_and_routing.py` + `tests/unit/test_control.py`) is implemented and verified. Week 6 Gateway lifecycle release, `RequestTrace` JSON serialization, and E2E/Kaggle verification belong to Person B as defined below.

## 1. Public Interfaces Implemented by Person A

### 1.1 `llm_serving_platform.control.registry`

- **`WorkerRecord(frozen=True)`**:
  ```python
  @dataclass(frozen=True)
  class WorkerRecord:
      worker_id: str
      endpoint: str
      model: str
      status: WorkerStatus = WorkerStatus.STARTING
      last_checked_at: datetime | None = None

      active_requests: int = 0
      last_load_change_at: datetime | None = None
  ```
  - Automatically serialized by `GET /control/v1/workers` and `GET /control/v1/workers/{worker_id}` (`control/api.py` requires no structural code changes, as its `response_model=list[WorkerRecord]` / `WorkerRecord` already serializes the two new fields).
- **`WorkerRoutingCandidate(frozen=True)`** (internal routing representation; never exposed on HTTP endpoints):
  ```python
  @dataclass(frozen=True)
  class WorkerRoutingCandidate:
      worker_id: str
      active_requests: int
      worker: InferenceWorker
  ```
- **Exceptions**:
  - `DuplicateRequestReservationError(ValueError)`: Raised by `reserve_request()` (and propagated out of `router.select_worker(trace)`) when `request_id` already holds an active reservation in `WorkerRegistry`.
  - `WorkerNotRoutableError(RuntimeError)`: Raised by `reserve_request()` when the target worker is currently not `WorkerStatus.HEALTHY`. Handled internally by `RoundRobinRouter` and `LeastLoadedRouter` with a bounded single refresh/retry before raising `WorkerUnavailableError(status_code=503)`.
- **`WorkerRegistry` Load & Reservation Methods**:
  - `routing_candidates() -> tuple[WorkerRoutingCandidate, ...]`: Returns a point-in-time snapshot of `HEALTHY` workers only, sorted deterministically by `worker_id`.
  - `reserve_request(worker_id: str, request_id: str, *, changed_at: datetime | None = None) -> int`: Atomically reserves `request_id` on `worker_id`, increments `active_requests`, updates `last_load_change_at` (UTC), and returns the new `active_requests` count.
  - `release_request(request_id: str, *, changed_at: datetime | None = None) -> bool`: Idempotently releases an active reservation for `request_id`. Returns `True` if an active reservation existed and was released (decrementing `active_requests` and updating `last_load_change_at`), or `False` if `request_id` was not actively reserved. Safe to call in `finally` blocks and defensive error handlers even if called multiple times or after a worker turns `UNHEALTHY`.
  - `active_request_count(worker_id: str) -> int`: Returns current `active_requests` for `worker_id` (or raises `UnknownWorkerError`).

### 1.2 `llm_serving_platform.common.config`

- `GatewayConfig.routing_policy` is validated and normalized in `GatewayConfig.__post_init__()` via `normalize_routing_policy(policy)`:
  - `"single"` / `"single_worker"` -> `"single"`
  - `"round_robin"` -> `"round_robin"`
  - `"least_loaded"` -> `"least_loaded"`
  - Any other value raises `ValueError` immediately at config/app creation time (fail-fast).

### 1.3 `llm_serving_platform.routing.router`

- **`create_routing_policy(config: GatewayConfig, registry: WorkerRegistry, single_worker: InferenceWorker | None = None) -> RoutingPolicy`**:
  - Maps `"single"` / `"single_worker"` -> `SingleWorkerRouter(single_worker)`
  - Maps `"round_robin"` -> `RoundRobinRouter(registry)`
  - Maps `"least_loaded"` -> `LeastLoadedRouter(registry)`
  - Raises `ValueError` on any unsupported policy.
- **Atomic Reservation Inside `select_worker(trace)`**:
  - Both `RoundRobinRouter.select_worker(trace)` and `LeastLoadedRouter.select_worker(trace)` acquire their policy selection lock (`self._lock`) BEFORE calling `WorkerRegistry` methods (`self._registry._lock`), enforcing strict lock ordering (`policy._lock -> registry._lock`).
  - A successful return from `router.select_worker(trace)` means the request reservation for `trace.request_id` is **already committed** in `WorkerRegistry`. **Gateway MUST NOT call `reserve_request()` a second time.**
- **Telemetry Attributes Populated on `trace: RequestTrace`**:
  - On every routing call, `RoundRobinRouter`, `LeastLoadedRouter`, and `SingleWorkerRouter` set these 5 attributes directly on `trace`:
    - `trace.routing_policy`: `"round_robin"` | `"least_loaded"` | `"single"`
    - `trace.routing_candidate_loads`: `dict[str, int]` mapping each `HEALTHY` candidate's `worker_id` to its `active_requests` **before** this reservation (e.g. `{"worker-vllm-0": 4, "worker-vllm-1": 1}`).
    - `trace.routing_selected_load_before`: `int | None` (load of selected worker before reservation).
    - `trace.routing_selected_load_after`: `int | None` (`selected_load_before + 1` after reservation).
    - `trace.routing_reason`: `"round_robin"` (for `RoundRobinRouter`), `"min_active_requests"` (for `LeastLoadedRouter`), `"single_worker"` (for `SingleWorkerRouter`), or `None` when routing fails with 503.

## 2. Person B Implementation Checklist (What Person B Needs to Wire)

### 2.1 `src/llm_serving_platform/telemetry/trace.py`
1. Declare the 5 routing fields on `RequestTrace` with safe defaults:
   ```python
   routing_policy: str | None = None
   routing_candidate_loads: dict[str, int] = field(default_factory=dict)
   routing_selected_load_before: int | None = None
   routing_selected_load_after: int | None = None
   routing_reason: str | None = None
   ```
2. In `RequestTrace.to_dict()`, retain all existing top-level keys (`"worker_id"`, `"milestones"`, `"metrics_ms"`, etc.) and add the `"routing"` dictionary per Contract Section 23:
   ```python
   "routing": {
       "policy": self.routing_policy,
       "candidate_loads": dict(self.routing_candidate_loads),
       "selected_worker_id": self.worker_id if self.worker_id and self.worker_id != "none" else None,
       "selected_load_before": self.routing_selected_load_before,
       "selected_load_after": self.routing_selected_load_after,
       "reason": self.routing_reason,
   },
   ```

### 2.2 `src/llm_serving_platform/gateway/main.py`
1. **Use `create_routing_policy` in `create_app()`**:
   Replace the `if cfg.routing_policy in ... else RoundRobinRouter(...)` block (lines 141–145) with:
   ```python
   app.state.router = create_routing_policy(
       cfg,
       active_registry,
       single_worker=app.state.worker,
   )
   ```
2. **Handle `DuplicateRequestReservationError` -> HTTP `409 Conflict` in `POST /v1/chat/completions`** (Contract Section 9):
   Catch `DuplicateRequestReservationError` around `router.select_worker(trace)` and return a `409 Conflict` `JSONResponse`:
   ```python
   except DuplicateRequestReservationError as exc:
       trace.mark_last_token(status="error", error=f"Duplicate active request_id: {exc}")
       trace_store.save(trace)
       return JSONResponse(
           status_code=409,
           content={
               "error": {
                   "message": f"Duplicate active request_id: {request_id}",
                   "code": 409,
               }
           },
           headers={"X-Request-ID": request_id, "X-Worker-ID": "none"},
       )
   ```
3. **Wrap `safe_event_generator` with Reservation Release (`finally`) and Guard Pre-Stream Errors** (Contract Sections 17, 18, 19):
   - Define a Gateway-level async generator wrapper:
     ```python
     async def _stream_with_reservation_release(
         registry: WorkerRegistry,
         *args,
         request_id: str,
         **kwargs,
     ):
         try:
             async for chunk in safe_event_generator(*args, request_id=request_id, **kwargs):
                 yield chunk
         finally:
             registry.release_request(request_id)
     ```
   - Wrap the post-routing setup in `try ... except`: if any exception occurs after `router.select_worker(trace)` succeeds and before `StreamingResponse` takes ownership of the generator, call `request.app.state.registry.release_request(request_id)` before re-raising.

### 2.3 Verification & Integration Tests Owned by Person B
- Add lifecycle release integration tests (normal completion, pre-stream worker error, mid-stream error, timeout, client cancellation, unexpected exception -> all `active_requests == 0` afterward).
- Add Control API active-request integration test (`active_requests >= 1` while slow request is in flight, returning to `0` on completion).
- Implement `infra/scripts/verify_week6_e2e.py` (`--no-gpu`, `--gateway-url`, `--model`, `--save-evidence`), register `"w6"` in `infra/scripts/verify.py`, and run the two-phase (`round_robin` vs `least_loaded`) Kaggle T4×2 harness.
