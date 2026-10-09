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
    background: str = "black"
    azimuth: float | None = None
    elevation: float | None = None
    distance: float | None = None
    zoom: float | None = None
    fov: float | None = None
    look_at: str | None = None
    view: int | None = None


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
        "background": request.background,
        "azimuth": request.azimuth,
        "elevation": request.elevation,
        "distance": request.distance,
        "zoom": request.zoom,
        "fov": request.fov,
        "look_at": request.look_at,
    }
    if request.view is not None:
        params["view"] = request.view
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
