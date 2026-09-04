import mcp.types as types

from splat.handlers.diffuse import DiffuseRequest, handle
from splat.mcp._content import image_content, text_content


def diffuse(
    prompt: str,
    model: str = "sdxl-turbo-mlx",
    negative_prompt: str = "",
    steps: int | None = None,
    seed: int | None = None,
    device: str = "auto",
) -> list[types.ContentBlock]:
    """Diffuse an image from a text prompt (cached; the pipeline's origin stage)."""
    result = handle(
        DiffuseRequest(
            prompt=prompt,
            model=model,
            negative_prompt=negative_prompt,
            steps=steps,
            seed=seed,
            device=device,
        )
    )
    content: list[types.ContentBlock] = [text_content(f"asset id: {result.asset.id}")]
    if result.license_warning:
        content.append(text_content(f"warning: {result.license_warning}"))
    content.append(image_content(result.asset))
    return content
