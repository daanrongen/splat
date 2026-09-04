from dataclasses import dataclass

from splat.application.pipeline import run_mesh
from splat.domain.asset import Asset
from splat.registry.wiring import get_asset_cache, get_mesh_backend


@dataclass(frozen=True)
class MeshRequest:
    inputs: list[Asset]
    model: str = "triposr"
    device: str = "auto"


def handle(request: MeshRequest) -> list[Asset]:
    cache = get_asset_cache()
    backend = get_mesh_backend(request.model, device=request.device)
    return [
        run_mesh(backend, cache, model_name=request.model, input_asset=asset, params={})
        for asset in request.inputs
    ]
