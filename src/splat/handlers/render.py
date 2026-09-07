from collections.abc import Callable
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
    engine: str = "cycles"


def handle(
    request: RenderRequest, *, on_progress: Callable[[str], None] | None = None
) -> list[Manifest]:
    validate_inputs(RENDER_CONTRACT, request.inputs)
    cache = get_manifest_repository()
    backend = get_render_backend(request.model)
    params = {
        "width": request.width,
        "height": request.height,
        "samples": request.samples,
        "engine": request.engine,
    }
    return [
        run_render(
            backend,
            cache,
            model_name=request.model,
            input_asset=asset,
            params=params,
            on_progress=on_progress,
        )
        for asset in request.inputs
    ]
