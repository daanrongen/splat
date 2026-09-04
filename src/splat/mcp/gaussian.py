from pathlib import Path

import mcp.types as types

from splat.handlers.gaussian import GaussianRequest
from splat.mcp._content import text_content
from splat.registry.wiring import get_client


def gaussian(
    images: list[str],
    output_path: str,
    model: str = "mvsplat",
    device: str = "auto",
) -> list[types.ContentBlock]:
    """Reconstruct a Gaussian splat from 2+ image paths, written to output_path."""
    result = get_client().gaussian(
        GaussianRequest(
            inputs=[Path(p) for p in images],
            output_path=Path(output_path),
            model=model,
            device=device,
        )
    )
    content = [text_content(f"wrote {output_path} ({result.cloud.point_count:,} points)")]
    content.extend(text_content(f"warning: {w}") for w in result.warnings)
    return content
