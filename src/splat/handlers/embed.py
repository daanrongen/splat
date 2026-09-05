from dataclasses import dataclass

from splat.application.pipeline import run_embed_image, run_embed_text
from splat.domain.contracts import validate_inputs
from splat.domain.errors import SplatDomainError
from splat.domain.manifest import Manifest, ManifestKind
from splat.registry.embedding import EMBED_IMAGE_CONTRACT
from splat.registry.wiring import get_embedding_backend, get_manifest_repository


@dataclass(frozen=True)
class EmbedRequest:
    inputs: list[Manifest] | None = None
    text: str | None = None
    model: str = "mobileclip2-s0"
    device: str = "auto"


def handle(request: EmbedRequest) -> list[Manifest]:
    has_inputs = bool(request.inputs)
    has_text = request.text is not None
    if has_inputs == has_text:
        raise SplatDomainError("embed requires either image input or --text, but not both.")
    if has_inputs:
        validate_inputs(EMBED_IMAGE_CONTRACT, request.inputs or [])

    cache = get_manifest_repository()
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
        if asset.kind == ManifestKind.CAPTION:
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
