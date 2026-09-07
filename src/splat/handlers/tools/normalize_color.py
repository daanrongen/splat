from dataclasses import dataclass

from splat.application.pipeline import run_normalize_color
from splat.domain.contracts import validate_inputs
from splat.domain.manifest import Manifest
from splat.registry.normalize_color import NORMALIZE_COLOR_CONTRACT
from splat.registry.wiring import get_manifest_repository


@dataclass(frozen=True)
class NormalizeColorRequest:
    inputs: list[Manifest]


def handle(request: NormalizeColorRequest) -> list[Manifest]:
    validate_inputs(NORMALIZE_COLOR_CONTRACT, request.inputs)
    cache = get_manifest_repository()
    return run_normalize_color(cache, input_assets=request.inputs, params={})
