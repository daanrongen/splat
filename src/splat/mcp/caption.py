import mcp.types as types

from splat.domain.asset import AssetKind
from splat.handlers.caption import DEFAULT_CAPTION_PROMPT, CaptionRequest
from splat.mcp._content import resource_content, text_content
from splat.mcp._inputs import resolve_input_asset
from splat.registry.wiring import get_asset_cache, get_client


def caption(
    image: str,
    model: str = "fastvlm-0.5b",
    prompt: str = DEFAULT_CAPTION_PROMPT,
    max_tokens: int = 80,
    temperature: float = 0.0,
    device: str = "auto",
) -> list[types.ContentBlock]:
    """Caption an image (path or @<asset-id>) as a UTF-8 text asset."""
    cache = get_asset_cache()
    asset = resolve_input_asset(image, cache, default_kind=AssetKind.IMAGE)
    result = get_client().caption(
        CaptionRequest(
            inputs=[asset],
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            device=device,
        )
    )[0]
    text = result.content_path.read_text(encoding="utf-8")
    return [text_content(text), text_content(f"asset id: {result.id}"), resource_content(result)]
