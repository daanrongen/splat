import mcp.types as types

from splat.domain.asset import AssetKind
from splat.handlers.segment import SegmentRequest, handle
from splat.mcp._content import image_content, text_content
from splat.mcp._inputs import resolve_input_asset
from splat.registry.wiring import get_asset_cache


def segment(
    image: str,
    model: str = "sam-mlx",
    max_stickers: int = 20,
    device: str = "auto",
) -> list[types.ContentBlock]:
    """Segment an image (path or @<asset-id>) into RGBA sticker cutouts."""
    cache = get_asset_cache()
    asset = resolve_input_asset(image, cache, default_kind=AssetKind.IMAGE)
    stickers = handle(
        SegmentRequest(inputs=[asset], model=model, max_stickers=max_stickers, device=device)
    )
    content: list[types.ContentBlock] = [
        text_content(f"{len(stickers)} stickers: " + ", ".join(s.id for s in stickers))
    ]
    content.extend(image_content(s) for s in stickers)
    return content
