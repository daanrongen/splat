from fastapi import APIRouter

from splat.application.models_admin import model_source_label, model_stage
from splat.handlers import models as models_handler
from splat.http._schemas import ModelInfoResponse, ModelSummary

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


@router.post("/models/{name}/pull")
def pull(name: str) -> dict:
    models_handler.pull(name)
    return {"pulled": name}


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
