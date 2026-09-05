import mcp.types as types

from splat.domain.manifest import ManifestKind
from splat.handlers.depth import DepthRequest
from splat.mcp._content import resource_content, text_content
from splat.mcp._inputs import resolve_input_asset
from splat.registry.wiring import get_client, get_manifest_repository


def depth(
    image: str,
    model: str = "depth-pro",
    device: str = "auto",
) -> list[types.ContentBlock]:
    """Estimate per-pixel metric depth for an image (path or @<asset-id>)."""
    cache = get_manifest_repository()
    asset = resolve_input_asset(image, cache, default_kind=ManifestKind.IMAGE)
    result = get_client().depth(DepthRequest(inputs=[asset], model=model, device=device))[0]
    return [text_content(f"asset id: {result.id}"), resource_content(result)]
