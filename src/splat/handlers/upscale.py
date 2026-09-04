from dataclasses import dataclass

from splat.application.pipeline import run_upscale
from splat.domain.asset import Asset, AssetKind
from splat.domain.errors import SplatDomainError
from splat.registry.wiring import get_asset_cache, get_upscale_backend


@dataclass(frozen=True)
class UpscaleRequest:
    inputs: list[Asset]
    model: str = "realesrgan-mlx"
    factor: int = 4
    tile: int = 0


def handle(request: UpscaleRequest) -> list[Asset]:
    cache = get_asset_cache()
    backend = get_upscale_backend(request.model)
    results = []
    for asset in request.inputs:
        if asset.kind not in (AssetKind.IMAGE, AssetKind.STICKER):
            raise SplatDomainError("upscale requires an image or sticker asset.")
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
