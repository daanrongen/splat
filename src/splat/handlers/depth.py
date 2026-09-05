from dataclasses import dataclass

from splat.application.pipeline import run_depth
from splat.domain.contracts import validate_inputs
from splat.domain.manifest import Manifest
from splat.registry.depth import DEPTH_CONTRACT
from splat.registry.wiring import get_asset_cache, get_depth_backend


@dataclass(frozen=True)
class DepthRequest:
    inputs: list[Manifest]
    model: str = "depth-pro"
    device: str = "auto"


def handle(request: DepthRequest) -> list[Manifest]:
    validate_inputs(DEPTH_CONTRACT, request.inputs)
    cache = get_asset_cache()
    backend = get_depth_backend(request.model, device=request.device)
    return [
        run_depth(backend, cache, model_name=request.model, input_asset=asset, params={})
        for asset in request.inputs
    ]
