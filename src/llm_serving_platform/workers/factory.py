"""Worker factory decoupling Gateway HTTP wiring from concrete worker runtime implementations."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Optional

import httpx

from llm_serving_platform.common.config import GatewayConfig
from llm_serving_platform.workers.base import InferenceWorker
from llm_serving_platform.workers.mock_adapter import MockInferenceWorker
from llm_serving_platform.workers.vllm_adapter import VLLMWorkerAdapter

if TYPE_CHECKING:
    from llm_serving_platform.control.registry import WorkerRegistry

DEFAULT_NO_GPU_CHUNKS = [
    "Continuous",
    " Batching",
    " ghép",
    " các",
    " request",
    " theo",
    " từng",
    " bước",
    " decode.",
]


def create_worker(
    config: GatewayConfig,
    client: Optional[httpx.AsyncClient] = None,
) -> InferenceWorker:
    """Instantiate the appropriate InferenceWorker based on GatewayConfig (Live GPU vs No-GPU)."""
    if config.no_gpu:
        worker_id = (
            config.worker_id
            if config.worker_id and config.worker_id != "worker-vllm-0"
            else "worker-mock-0"
        )
        return MockInferenceWorker(
            canned_chunks=list(DEFAULT_NO_GPU_CHUNKS),
            chunk_delay_s=0.01,
            worker_id=worker_id,
        )
    return VLLMWorkerAdapter(
        base_url=config.vllm_base_url,
        client=client,
        timeout=config.worker_timeout_s,
        worker_id=config.worker_id,
    )


def populate_registry(
    registry: WorkerRegistry,
    config: GatewayConfig,
    client: Optional[httpx.AsyncClient] = None,
) -> list[InferenceWorker]:
    """Register configured worker pool in WorkerRegistry by reusing create_worker."""
    if config.no_gpu:
        specs = [
            (f"worker-mock-{i}", f"mock://worker-mock-{i}")
            for i in range(max(1, config.mock_worker_count))
        ]
    elif config.worker_endpoints:
        specs = [
            tuple(ep.split("=", 1)) if "=" in ep else (f"worker-vllm-{i}", ep)
            for i, ep in enumerate(config.worker_endpoints)
        ]
    else:
        specs = [(config.worker_id, config.vllm_base_url)]

    workers: list[InferenceWorker] = []
    for w_id, endpoint in specs:
        worker_cfg = replace(config, worker_id=w_id.strip(), vllm_base_url=endpoint.strip())
        worker = create_worker(config=worker_cfg, client=client)
        registry.register(worker, endpoint.strip(), config.model_name)
        if config.no_gpu:
            registry.record_health(worker.worker_id, True)
        workers.append(worker)
    return workers
