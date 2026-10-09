import mcp.types as types

from splat.domain.manifest import ManifestKind
from splat.handlers.segment import SegmentRequest
from splat.mcp._content import image_content, text_content
from splat.mcp._inputs import resolve_input_asset
from splat.registry.wiring import get_client, get_manifest_repository


def segment(
    image: str,
    model: str = "sam-mlx",
    max_stickers: int = 20,
    device: str = "auto",
    foreground: bool = False,
    drop_background: bool = False,
) -> list[types.ContentBlock]:
    """Segment an image (path or @<asset-id>) into RGBA sticker cutouts."""
    cache = get_manifest_repository()
    asset = resolve_input_asset(image, cache, default_kind=ManifestKind.IMAGE)
    stickers = get_client().segment(
        SegmentRequest(
            inputs=[asset],
            model=model,
            max_stickers=max_stickers,
            device=device,
            foreground=foreground,
            drop_background=drop_background,
        )
    )
    content: list[types.ContentBlock] = [
        text_content(f"{len(stickers)} stickers: " + ", ".join(s.id for s in stickers))
    ]
    content.extend(image_content(s) for s in stickers)
    return content
