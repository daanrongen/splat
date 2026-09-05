import mcp.types as types

from splat.domain.errors import SplatDomainError
from splat.domain.manifest import ManifestKind
from splat.handlers.embed import EmbedRequest
from splat.mcp._content import resource_content, text_content
from splat.mcp._inputs import resolve_input_asset
from splat.registry.wiring import get_client, get_manifest_repository


def embed(
    image: str | None = None,
    text: str | None = None,
    model: str = "mobileclip2-s0",
    device: str = "auto",
) -> list[types.ContentBlock]:
    """Embed an image path/@asset-id or text string as a normalized vector asset."""
    if (image is None) == (text is None):
        raise SplatDomainError("embed requires either image or text, but not both.")

    cache = get_manifest_repository()
    if text is not None:
        result = get_client().embed(EmbedRequest(text=text, model=model, device=device))[0]
    else:
        asset = resolve_input_asset(image or "", cache, default_kind=ManifestKind.IMAGE)
        result = get_client().embed(EmbedRequest(inputs=[asset], model=model, device=device))[0]

    return [
        text_content(
            f"asset id: {result.id}; {result.metadata.input_type} "
            f"{result.metadata.dimension}d {result.metadata.dtype}"
        ),
        resource_content(result),
    ]
