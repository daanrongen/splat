import mcp.types as types

from splat.handlers.tools import DisplaceHeightRequest, handle
from splat.mcp._content import resource_content, text_content
from splat.registry.wiring import get_asset_cache


def displace_height(depth_asset_id: str, export_format: str = "glb") -> list[types.ContentBlock]:
    """Displace a depth map (the asset id from a prior `depth` call) into a
    triangulated, textured mesh."""
    cache = get_asset_cache()
    depth_asset = cache.get(depth_asset_id)
    result = handle(DisplaceHeightRequest(inputs=[depth_asset], export_format=export_format))[0]
    return [text_content(f"asset id: {result.id}"), resource_content(result)]
