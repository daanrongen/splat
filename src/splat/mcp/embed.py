import mcp.types as types

from splat.domain.asset import AssetKind
from splat.domain.errors import SplatDomainError
from splat.handlers.embed import EmbedRequest
from splat.mcp._content import resource_content, text_content
from splat.mcp._inputs import resolve_input_asset
from splat.registry.wiring import get_asset_cache, get_client


def embed(
    image: str | None = None,
    text: str | None = None,
    model: str = "mobileclip2-s0",
    device: str = "auto",
) -> list[types.ContentBlock]:
    """Embed an image path/@asset-id or text string as a normalized vector asset."""
    if (image is None) == (text is None):
        raise SplatDomainError("embed requires either image or text, but not both.")

    cache = get_asset_cache()
    if text is not None:
        result = get_client().embed(EmbedRequest(text=text, model=model, device=device))[0]
    else:
        asset = resolve_input_asset(image or "", cache, default_kind=AssetKind.IMAGE)
        result = get_client().embed(EmbedRequest(inputs=[asset], model=model, device=device))[0]

    return [
        text_content(
            f"asset id: {result.id}; {result.metadata.get('input_type')} "
            f"{result.metadata.get('dimension')}d {result.metadata.get('dtype')}"
        ),
        resource_content(result),
    ]
