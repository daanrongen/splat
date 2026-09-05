from dataclasses import dataclass

from splat.application.pipeline import run_upscale
from splat.domain.contracts import validate_inputs
from splat.domain.manifest import Manifest
from splat.registry.upscale import UPSCALE_CONTRACT
from splat.registry.wiring import get_manifest_repository, get_upscale_backend


@dataclass(frozen=True)
class UpscaleRequest:
    inputs: list[Manifest]
    model: str = "realesrgan-mlx"
    factor: int = 4
    tile: int = 0


def handle(request: UpscaleRequest) -> list[Manifest]:
    validate_inputs(UPSCALE_CONTRACT, request.inputs)
    cache = get_manifest_repository()
    backend = get_upscale_backend(request.model)
    results = []
    for asset in request.inputs:
        results.append(
            run_upscale(
                backend,
                cache,
                model_name=request.model,
                input_asset=asset,
                params={"factor": request.factor, "tile": request.tile},
            )
        )
    return results
