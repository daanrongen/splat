from dataclasses import dataclass

from splat.application.pipeline import drop_background, run_foreground, run_segment
from splat.domain.contracts import validate_inputs
from splat.domain.manifest import Manifest
from splat.registry.segment import SEGMENT_CONTRACT
from splat.registry.wiring import get_manifest_repository, get_segmentation_backend


@dataclass(frozen=True)
class SegmentRequest:
    inputs: list[Manifest]
    model: str = "sam-mlx"
    max_stickers: int = 20
    device: str = "auto"
    foreground: bool = False
    drop_background: bool = False


def handle(request: SegmentRequest) -> list[Manifest]:
    validate_inputs(SEGMENT_CONTRACT, request.inputs)
    cache = get_manifest_repository()
    backend = get_segmentation_backend(request.model, device=request.device)
    all_stickers: list[Manifest] = []
    for asset in request.inputs:
        stickers = run_segment(
            backend,
            cache,
            model_name=request.model,
            input_asset=asset,
            params={"max_stickers": request.max_stickers},
        )
        if request.foreground:
            stickers = [run_foreground(cache, input_asset=asset, stickers=stickers)]
        elif request.drop_background:
            stickers = drop_background(asset, stickers)
        all_stickers.extend(stickers)
    return all_stickers
