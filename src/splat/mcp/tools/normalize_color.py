import mcp.types as types

from splat.handlers.tools.normalize_color import NormalizeColorRequest, handle
from splat.mcp._content import resource_content, text_content
from splat.registry.wiring import get_manifest_repository


def normalize_color(image_asset_ids: list[str]) -> list[types.ContentBlock]:
    """Correct per-view exposure/white-balance drift across a multi-photo capture."""
    cache = get_manifest_repository()
    assets = [cache.get(asset_id) for asset_id in image_asset_ids]
    results = handle(NormalizeColorRequest(inputs=assets))
    content: list[types.ContentBlock] = [text_content(f"corrected {len(results)} images")]
    content.extend(resource_content(result) for result in results)
    return content
