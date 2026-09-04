from dataclasses import dataclass

from splat.application.pipeline import run_segment
from splat.domain.asset import Asset
from splat.registry.wiring import get_asset_cache, get_segmentation_backend


@dataclass(frozen=True)
class SegmentRequest:
    inputs: list[Asset]
    model: str = "sam-mlx"
    max_stickers: int = 20
    device: str = "auto"


def handle(request: SegmentRequest) -> list[Asset]:
    cache = get_asset_cache()
    backend = get_segmentation_backend(request.model, device=request.device)
    all_stickers: list[Asset] = []
    for asset in request.inputs:
        all_stickers.extend(
            run_segment(
                backend,
                cache,
                model_name=request.model,
                input_asset=asset,
                params={"max_stickers": request.max_stickers},
            )
        )
    return all_stickers
