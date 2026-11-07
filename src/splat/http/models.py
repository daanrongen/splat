import asyncio
import threading
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from splat.application.models_admin import model_source_label, model_stage
from splat.domain.errors import SplatDomainError
from splat.handlers import models as models_handler
from splat.http._schemas import ModelInfoResponse, ModelSummary, PullEvent

router = APIRouter()


@router.get("/models")
def list_models() -> list[ModelSummary]:
    rows = models_handler.list_models()
    return [
        ModelSummary(
            name=descriptor.name,
            stage=model_stage(descriptor.name),
            runtime=descriptor.runtime,
            license=descriptor.license.spdx_id,
            commercial=descriptor.license.is_commercial,
            cached=cached,
        )
        for descriptor, cached in rows
    ]


class _ClientGone(Exception):
    """Raised inside the download once the client disconnected, to stop it."""


async def _pull_events(name: str) -> AsyncIterator[str]:
    loop = asyncio.get_running_loop()
    events: asyncio.Queue[PullEvent] = asyncio.Queue()
    gone = threading.Event()
    last_sent = 0.0

    def emit(event: PullEvent) -> None:
        loop.call_soon_threadsafe(events.put_nowait, event)

    def on_progress(repo_id: str, done: int, total: int) -> None:
        nonlocal last_sent
        if gone.is_set():
            raise _ClientGone
        if done >= total or time.monotonic() - last_sent >= 0.1:
            last_sent = time.monotonic()
            emit(PullEvent(repo=repo_id, done=done, total=total))

    def run() -> None:
        try:
            models_handler.pull(name, on_progress=on_progress)
            emit(PullEvent(pulled=name))
        except _ClientGone:
            pass
        except SplatDomainError as exc:
            emit(PullEvent(code="domain_error", detail=str(exc)))
        except Exception as exc:
            emit(PullEvent(code="internal_error", detail=f"{type(exc).__name__}: {exc}"))

    loop.run_in_executor(None, run)
    try:
        while True:
            event = await events.get()
            yield event.model_dump_json(exclude_none=True) + "\n"
            if event.pulled or event.code:
                return
    finally:
        gone.set()


@router.post("/models/{name}/pull")
def pull(name: str) -> StreamingResponse:
    models_handler.info(name)  # an unknown name is a 422 before the stream starts
    return StreamingResponse(_pull_events(name), media_type="application/x-ndjson")


@router.get("/models/{name}")
def info(name: str) -> ModelInfoResponse:
    descriptor = models_handler.info(name)
    return ModelInfoResponse(
        name=descriptor.name,
        runtime=descriptor.runtime,
        source=model_source_label(descriptor),
        license=str(descriptor.license),
        min_images=descriptor.min_images,
        max_images=descriptor.max_images,
        dimension=descriptor.dimension,
        normalized=descriptor.normalized,
        notes=descriptor.notes,
    )


@router.delete("/models/{name}")
def rm(name: str) -> dict:
    models_handler.rm(name)
    return {"removed": name}
