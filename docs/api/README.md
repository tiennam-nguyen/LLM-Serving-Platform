# API Documentation

This directory contains specifications and contract definitions for all external and internal API interfaces of the **Adaptive LLM Serving Platform**.

---

## Intended API Surfaces

### 1. Client-Facing Serving API (Data Plane)
- **Standard**: OpenAI-compatible chat completion endpoints (e.g., `POST /v1/chat/completions`).
- **Protocols**: JSON request payloads and Server-Sent Events (SSE) for streaming token responses.
- **Contract Boundary**: Handled by `llm_serving_platform.gateway`.

### 2. Internal Worker Adapter Interface
- **Responsibility**: Standardized Python protocol/interface between `llm_serving_platform.routing` / `gateway` and underlying inference runtimes (`llm_serving_platform.workers`).
- **Contract**: Decouples engine-specific payload formats and lifecycle management.

### 3. Control Plane Management API (Future)
- **Responsibility**: Administrative operations, worker health probes, and cluster topology queries.
- **Contract Boundary**: Handled by `llm_serving_platform.control`.
