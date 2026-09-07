from dataclasses import dataclass

from splat.application.pipeline import run_render
from splat.domain.contracts import validate_inputs
from splat.domain.manifest import Manifest
from splat.registry.render import RENDER_CONTRACT
from splat.registry.wiring import get_manifest_repository, get_render_backend


@dataclass(frozen=True)
class RenderRequest:
    inputs: list[Manifest]
    model: str = "blender"
    width: int = 1280
    height: int = 720
    samples: int = 32


def handle(request: RenderRequest) -> list[Manifest]:
    validate_inputs(RENDER_CONTRACT, request.inputs)
    cache = get_manifest_repository()
    backend = get_render_backend(request.model)
    params = {"width": request.width, "height": request.height, "samples": request.samples}
    return [
        run_render(backend, cache, model_name=request.model, input_asset=asset, params=params)
        for asset in request.inputs
    ]
