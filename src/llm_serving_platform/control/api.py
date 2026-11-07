"""Read-only router factory; mounting and CORS policy belong to the gateway owner."""

from fastapi import APIRouter, HTTPException

from llm_serving_platform.control.registry import (
    UnknownWorkerError,
    WorkerRecord,
    WorkerRegistry,
)


def create_control_router(registry: WorkerRegistry) -> APIRouter:
    router = APIRouter(prefix="/control/v1/workers", tags=["workers"])

    @router.get("", response_model=list[WorkerRecord])
    async def list_workers() -> list[WorkerRecord]:
        return registry.list_workers()

    @router.get("/{worker_id}", response_model=WorkerRecord)
    async def get_worker(worker_id: str) -> WorkerRecord:
        try:
            return registry.get_record(worker_id)
        except UnknownWorkerError:
            raise HTTPException(status_code=404, detail="Worker not found") from None

    return router
