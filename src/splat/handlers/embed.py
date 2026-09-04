from dataclasses import dataclass

from splat.application.pipeline import run_embed_image, run_embed_text
from splat.domain.asset import Asset, AssetKind
from splat.domain.errors import SplatDomainError
from splat.registry.wiring import get_asset_cache, get_embedding_backend


@dataclass(frozen=True)
class EmbedRequest:
    inputs: list[Asset] | None = None
    text: str | None = None
    model: str = "mobileclip2-s0"
    device: str = "auto"


def handle(request: EmbedRequest) -> list[Asset]:
    has_inputs = bool(request.inputs)
    has_text = request.text is not None
    if has_inputs == has_text:
        raise SplatDomainError("embed requires either image input or --text, but not both.")

    cache = get_asset_cache()
    backend = get_embedding_backend(request.model, device=request.device)
    params = {"device": request.device}

    if has_text:
        return [
            run_embed_text(
                backend,
                cache,
                model_name=request.model,
                text=request.text or "",
                params=params,
            )
        ]

    results = []
    for asset in request.inputs or []:
        if asset.kind == AssetKind.CAPTION:
            text = asset.content_path.read_text(encoding="utf-8")
            results.append(
                run_embed_text(
                    backend,
                    cache,
                    model_name=request.model,
                    text=text,
                    params=params,
                    input_asset=asset,
                )
            )
            continue

        if asset.kind not in (AssetKind.IMAGE, AssetKind.STICKER):
            raise SplatDomainError("embed requires an image, sticker, or caption asset.")
        results.append(
            run_embed_image(
                backend,
                cache,
                model_name=request.model,
                input_asset=asset,
                params=params,
            )
        )
    return results
