from dataclasses import dataclass

from splat.application.pipeline import run_blender
from splat.domain.contracts import validate_inputs
from splat.domain.manifest import Manifest
from splat.registry.render import BLENDER_RENDER_CONTRACT
from splat.registry.wiring import get_manifest_repository, get_render_backend


@dataclass(frozen=True)
class BlenderRequest:
    inputs: list[Manifest]
    width: int = 1280
    height: int = 720
    samples: int = 32


def handle(request: BlenderRequest) -> list[Manifest]:
    validate_inputs(BLENDER_RENDER_CONTRACT, request.inputs)
    cache = get_manifest_repository()
    backend = get_render_backend()
    params = {"width": request.width, "height": request.height, "samples": request.samples}
    return [
        run_blender(backend, cache, input_asset=asset, params=params) for asset in request.inputs
    ]
