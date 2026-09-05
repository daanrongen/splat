import mcp.types as types

from splat.domain.manifest import ManifestKind
from splat.handlers.upscale import UpscaleRequest
from splat.mcp._content import resource_content, text_content
from splat.mcp._inputs import resolve_input_asset
from splat.registry.wiring import get_asset_cache, get_client


def upscale(
    image: str,
    model: str = "realesrgan-mlx",
    factor: int = 4,
    tile: int = 0,
) -> list[types.ContentBlock]:
    """Upscale an image or sticker asset with a super-resolution backend."""
    cache = get_asset_cache()
    asset = resolve_input_asset(image, cache, default_kind=ManifestKind.IMAGE)
    result = get_client().upscale(
        UpscaleRequest(inputs=[asset], model=model, factor=factor, tile=tile)
    )[0]
    return [text_content(f"asset id: {result.id}"), resource_content(result)]
