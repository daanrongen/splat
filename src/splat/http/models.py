from fastapi import APIRouter

from splat.handlers import models as models_handler
from splat.http._schemas import ModelSummary

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
def info(name: str) -> ModelSummary:
    descriptor = models_handler.info(name)
    return ModelSummary(
        name=descriptor.name,
        runtime=getattr(descriptor, "runtime", "-"),
        license=str(descriptor.license),
    )


@router.delete("/models/{name}")
def rm(name: str) -> dict:
    models_handler.rm(name)
    return {"removed": name}
