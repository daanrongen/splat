from dataclasses import dataclass, field

from splat.application.pipeline import run_diffuse
from splat.domain.contracts import validate_inputs
from splat.domain.manifest import Manifest
from splat.registry.diffuse import DIFFUSE_CONTRACT
from splat.registry.wiring import get_diffusion_backend, get_manifest_repository


@dataclass(frozen=True)
class DiffuseRequest:
    prompt: str
    inputs: list[Manifest] = field(default_factory=list)
    model: str = "sdxl-turbo-mlx"
    negative_prompt: str = ""
    steps: int | None = None
    seed: int | None = None
    strength: float | None = None
    width: int | None = None
    height: int | None = None
    device: str = "auto"


@dataclass(frozen=True)
class DiffuseResult:
    asset: Manifest
    license_warning: str | None


def handle(request: DiffuseRequest) -> DiffuseResult:
    validate_inputs(DIFFUSE_CONTRACT, request.inputs)
    backend = get_diffusion_backend(request.model, device=request.device)
    warning = (
        None if backend.license.is_commercial else f"{request.model} license: {backend.license}"
    )
    asset = run_diffuse(
        backend,
        get_manifest_repository(),
        model_name=request.model,
        prompt=request.prompt,
        input_asset=request.inputs[0] if request.inputs else None,
        params={
            "negative_prompt": request.negative_prompt,
            "steps": request.steps,
            "seed": request.seed,
            "strength": request.strength,
            "width": request.width,
            "height": request.height,
        },
    )
    return DiffuseResult(asset=asset, license_warning=warning)
