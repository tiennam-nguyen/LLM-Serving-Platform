# Integration Tests

This directory will host end-to-end and multi-module integration tests as features are developed in future milestones.

---

## Future Scope

1. **Gateway <-> Router Integration**: Testing HTTP request parsing, routing decision dispatch, and streaming response proxying using mock worker adapters.
2. **Worker Adapter <-> vLLM Serving**: Testing connectivity, health checking, and inference streaming against real or mock vLLM runtime endpoints.
3. **Telemetry & Metrics Verification**: Ensuring latency (TTFT/ITL) and token counts are accurately propagated to metrics sinks during active serving runs.

---

## Test Execution

Integration tests will be executed via `pytest`:

```bash
python -m pytest tests/integration
```
