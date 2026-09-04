from dataclasses import dataclass

from splat.application.pipeline import run_diffuse
from splat.domain.asset import Asset
from splat.registry.wiring import get_asset_cache, get_diffusion_backend


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
    asset: Asset
    license_warning: str | None


def handle(request: DiffuseRequest) -> DiffuseResult:
    backend = get_diffusion_backend(request.model, device=request.device)
    warning = (
        None if backend.license.is_commercial else f"{request.model} license: {backend.license}"
    )
    asset = run_diffuse(
        backend,
        get_asset_cache(),
        model_name=request.model,
        prompt=request.prompt,
        params={
            "negative_prompt": request.negative_prompt,
            "steps": request.steps,
            "seed": request.seed,
        },
    )
    return DiffuseResult(asset=asset, license_warning=warning)
