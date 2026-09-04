import mcp.types as types

from splat.domain.asset import AssetKind
from splat.handlers.mesh import MeshRequest, handle
from splat.mcp._content import resource_content, text_content
from splat.mcp._inputs import resolve_input_asset
from splat.registry.wiring import get_asset_cache


def mesh(
    image: str,
    model: str = "triposr",
    device: str = "auto",
) -> list[types.ContentBlock]:
    """Predict a 3D mesh from an image (path or @<asset-id>) using a learned model."""
    cache = get_asset_cache()
    asset = resolve_input_asset(image, cache, default_kind=AssetKind.IMAGE)
    result = handle(MeshRequest(inputs=[asset], model=model, device=device))[0]
    return [text_content(f"asset id: {result.id}"), resource_content(result)]
