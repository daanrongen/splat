from dataclasses import dataclass

from splat.application.pipeline import run_caption
from splat.domain.asset import Asset
from splat.registry.wiring import get_asset_cache, get_caption_backend

DEFAULT_CAPTION_PROMPT = "Describe this image in one concise sentence."


@dataclass(frozen=True)
class CaptionRequest:
    inputs: list[Asset]
    model: str = "fastvlm-0.5b"
    prompt: str = DEFAULT_CAPTION_PROMPT
    max_tokens: int = 80
    temperature: float = 0.0
    device: str = "auto"


def handle(request: CaptionRequest) -> list[Asset]:
    cache = get_asset_cache()
    backend = get_caption_backend(request.model, device=request.device)
    params = {
        "prompt": request.prompt,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
    }
    return [
        run_caption(backend, cache, model_name=request.model, input_asset=asset, params=params)
        for asset in request.inputs
    ]
