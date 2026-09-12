import mcp.types as types

from splat.domain.manifest import ManifestKind
from splat.handlers.diffuse import DiffuseRequest
from splat.mcp._content import image_content, text_content
from splat.mcp._inputs import resolve_input_asset
from splat.registry.wiring import get_client, get_manifest_repository


def diffuse(
    prompt: str,
    image: str | None = None,
    model: str = "sdxl-turbo-mlx",
    negative_prompt: str = "",
    steps: int | None = None,
    strength: float | None = None,
    seed: int | None = None,
    device: str = "auto",
) -> list[types.ContentBlock]:
    """Diffuse an image from a text prompt, or edit an existing image (image-to-image)
    when `image` (a path or `@<asset-id>`) is given."""
    inputs = []
    if image is not None:
        cache = get_manifest_repository()
        inputs = [resolve_input_asset(image, cache, default_kind=ManifestKind.IMAGE)]

    result = get_client().diffuse(
        DiffuseRequest(
            prompt=prompt,
            inputs=inputs,
            model=model,
            negative_prompt=negative_prompt,
            steps=steps,
            strength=strength,
            seed=seed,
            device=device,
        )
    )
    content: list[types.ContentBlock] = [text_content(f"asset id: {result.asset.id}")]
    if result.license_warning:
        content.append(text_content(f"warning: {result.license_warning}"))
    content.append(image_content(result.asset))
    return content
