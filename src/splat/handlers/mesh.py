from dataclasses import dataclass

from splat.application.pipeline import run_mesh
from splat.domain.contracts import validate_inputs
from splat.domain.manifest import Manifest
from splat.registry.mesh import MESH_CONTRACT
from splat.registry.wiring import get_manifest_repository, get_mesh_backend


@dataclass(frozen=True)
class MeshRequest:
    inputs: list[Manifest]
    model: str = "triposr"
    device: str = "auto"


def handle(request: MeshRequest) -> list[Manifest]:
    validate_inputs(MESH_CONTRACT, request.inputs)
    cache = get_manifest_repository()
    backend = get_mesh_backend(request.model, device=request.device)
    return [
        run_mesh(backend, cache, model_name=request.model, input_asset=asset, params={})
        for asset in request.inputs
    ]
