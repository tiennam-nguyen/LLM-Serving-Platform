"""Worker factory decoupling Gateway HTTP wiring from concrete worker runtime implementations."""

from typing import Optional

import httpx

from llm_serving_platform.common.config import GatewayConfig
from llm_serving_platform.workers.base import InferenceWorker
from llm_serving_platform.workers.mock_adapter import MockInferenceWorker
from llm_serving_platform.workers.vllm_adapter import VLLMWorkerAdapter

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
