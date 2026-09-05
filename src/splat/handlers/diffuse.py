from dataclasses import dataclass

from splat.application.pipeline import run_diffuse
from splat.domain.manifest import Manifest
from splat.registry.wiring import get_diffusion_backend, get_manifest_repository


@dataclass(frozen=True)
class DiffuseRequest:
    prompt: str
    model: str = "sdxl-turbo-mlx"
    negative_prompt: str = ""
    steps: int | None = None
    seed: int | None = None
    device: str = "auto"


@dataclass(frozen=True)
class DiffuseResult:
    asset: Manifest
    license_warning: str | None


def handle(request: DiffuseRequest) -> DiffuseResult:
    backend = get_diffusion_backend(request.model, device=request.device)
    warning = (
        None if backend.license.is_commercial else f"{request.model} license: {backend.license}"
    )
    asset = run_diffuse(
        backend,
        get_manifest_repository(),
        model_name=request.model,
        prompt=request.prompt,
        params={
            "negative_prompt": request.negative_prompt,
            "steps": request.steps,
            "seed": request.seed,
        },
    )
    return DiffuseResult(asset=asset, license_warning=warning)
