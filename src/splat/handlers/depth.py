from dataclasses import dataclass

from splat.application.pipeline import run_depth
from splat.domain.asset import Asset
from splat.registry.wiring import get_asset_cache, get_depth_backend


@dataclass(frozen=True)
class DepthRequest:
    inputs: list[Asset]
    model: str = "depth-pro"
    device: str = "auto"


def handle(request: DepthRequest) -> list[Asset]:
    cache = get_asset_cache()
    backend = get_depth_backend(request.model, device=request.device)
    return [
        run_depth(backend, cache, model_name=request.model, input_asset=asset, params={})
        for asset in request.inputs
    ]
