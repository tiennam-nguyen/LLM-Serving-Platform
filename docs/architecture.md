# System Architecture & Design Intent

This document outlines the high-level architecture, design principles, and component boundaries for the **Adaptive LLM Serving Platform**.

---

## 1. Architectural Principles

1. **Modular Monolith First**:
   The platform is structured as a cohesive Python package with explicit, decoupled module boundaries (`gateway`, `control`, `routing`, `workers`, `telemetry`, `benchmark`, `common`). We avoid premature distributed microservices or multi-repo fragmentation.

2. **Decoupled Data Plane and Control Plane**:
   The **data plane** (the fast path handling live inference requests) must never perform synchronous database queries or heavy external lookups. Routing decisions are made strictly in-memory using lightweight state snapshots.

3. **Pluggable Routing Policies**:
   Routing logic is separated from HTTP transport. The router consumes incoming request characteristics and in-memory worker state snapshots to return a chosen worker target without hardcoding routing algorithms into HTTP request handlers.

4. **Runtime-Agnostic Worker Adapters**:
   Serving runtimes (initially vLLM) sit behind a standard worker adapter interface. The core gateway and router communicate with adapters rather than binding to runtime-specific HTTP endpoints.

5. **Reproducibility by Default**:
   All benchmark runs and experimental evaluations record precise environment manifests: git commit hash, hardware topology, model identifier, routing policy, arrival process, random seed, and generation hyperparameters.

---

## 2. Component Topology

### Request Data Path vs. Control Path

```text
+-----------------------+       +-------------------------+
|   Web UI / Client     |       | Benchmark Runner Client |
+-----------+-----------+       +------------+------------+
            |                                |
            +----------------+---------------+
                             |
                             v
           +-----------------------------------+
           |        Serving Gateway            | (Data Plane: Protocol termination,
           +-----------------+-----------------+  streaming, validation)
                             |
                             v
           +-----------------------------------+
           |         Router Engine             | (Evaluates RoutingPolicy in-memory)
           +-----------------+-----------------+
                             |
            +----------------+----------------+
            |                                 |
            v                                 v
+-----------------------+         +-----------------------+
|  Worker Adapter #1    |         |  Worker Adapter #2    | (Encapsulates engine protocol)
+-----------+-----------+         +-----------+-----------+
            |                                 |
            v                                 v
+-----------------------+         +-----------------------+
|   vLLM Instance #1    |         |   vLLM Instance #2    | (GPU Serving Runtimes)
+-----------------------+         +-----------------------+
```

### Telemetry & In-Memory Routing Feedback Loop

```text
+-------------------------+         +-------------------------+
|    vLLM Instance #1     |         |    vLLM Instance #2     |
+------------+------------+         +------------+------------+
             |                                   |
             v                                   v
+-------------------------------------------------------------+
|                     Telemetry Subsystem                     |
|  (Collects TTFT, ITL, queue depth, GPU utilization metrics) |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|              In-Memory State Snapshot Buffer                |
|       (Atomic, thread-safe, fast in-memory worker state)    |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                     Router Engine                           |
|       (Consumes state snapshot to route next request)       |
+-------------------------------------------------------------+
```

---

## 3. Module Boundaries & Responsibilities

| Module | Boundary Role | Key Constraints |
| :--- | :--- | :--- |
| `gateway` | Data-plane entry point | Handles client HTTP/SSE streaming; no business routing logic. |
| `control` | Control-plane management | Manages cluster configuration and worker topology; out of the fast path. |
| `routing` | Policy execution | Pure algorithmic routing (`RoundRobin`, `LeastLoaded`, etc.); no direct I/O. |
| `workers` | Serving engine abstraction | Standardizes API communication with vLLM / mock runtimes. |
| `telemetry` | Metrics and observability | Collects latency, token rates, and GPU state into fast in-memory snapshots. |
| `benchmark` | Workload replay | Generates synthetic / trace workloads and records latency profiles. |
| `common` | Shared cross-cutting types | Minimal utility types, base models, and shared constants. |

---

## 4. Current Status

At initial repository bootstrap, these boundaries are represented as package structures with module docstrings. Implementation begins with Week 1 (single-GPU vLLM baseline).
