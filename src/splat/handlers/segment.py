from dataclasses import dataclass

from splat.application.pipeline import run_segment
from splat.domain.contracts import validate_inputs
from splat.domain.manifest import Manifest
from splat.registry.segmentation import SEGMENT_CONTRACT
from splat.registry.wiring import get_asset_cache, get_segmentation_backend


@dataclass(frozen=True)
class SegmentRequest:
    inputs: list[Manifest]
    model: str = "sam-mlx"
    max_stickers: int = 20
    device: str = "auto"


def handle(request: SegmentRequest) -> list[Manifest]:
    validate_inputs(SEGMENT_CONTRACT, request.inputs)
    cache = get_asset_cache()
    backend = get_segmentation_backend(request.model, device=request.device)
    all_stickers: list[Manifest] = []
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
