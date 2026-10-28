"""Telemetry module boundary.

Captures per-request 6-milestone lifecycle traces (arrival, routing_start, routing_end,
worker_start, first_token, last_token) and derived server latencies.
"""

from llm_serving_platform.telemetry.trace import InMemoryTraceStore, RequestTrace

__all__ = ["InMemoryTraceStore", "RequestTrace"]
