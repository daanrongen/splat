from fastapi import APIRouter

from splat.application.models_admin import model_source_label
from splat.handlers import models as models_handler
from splat.http._schemas import ModelInfoResponse, ModelSummary

router = APIRouter()


@router.get("/models")
def list_models() -> list[ModelSummary]:
    rows = models_handler.list_models()
    return [
        ModelSummary(
            name=descriptor.name,
            runtime=getattr(descriptor, "runtime", "-"),
            license=str(descriptor.license),
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
        runtime=getattr(descriptor, "runtime", "-"),
        source=model_source_label(descriptor),
        license=str(descriptor.license),
        min_images=getattr(descriptor, "min_images", None),
        max_images=getattr(descriptor, "max_images", None),
        dimension=getattr(descriptor, "dimension", None),
        normalized=getattr(descriptor, "normalized", None),
        notes=getattr(descriptor, "notes", ""),
    )


@router.delete("/models/{name}")
def rm(name: str) -> dict:
    models_handler.rm(name)
    return {"removed": name}
