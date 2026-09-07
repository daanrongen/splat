from pathlib import Path

import mcp.types as types

from splat.handlers.tools.extract_surface import ExtractSurfaceRequest
from splat.mcp._content import text_content
from splat.registry.wiring import get_client


def extract_surface(
    input_path: str,
    output_path: str,
    to: str | None = None,
    depth: int = 9,
    opacity_threshold: float = 0.1,
) -> list[types.ContentBlock]:
    result = get_client().tools_extract_surface(
        ExtractSurfaceRequest(
            input_path=Path(input_path),
            output_path=Path(output_path),
            format=to,
            depth=depth,
            opacity_threshold=opacity_threshold,
        )
    )
    return [
        text_content(
            f"wrote {output_path} ({result.vertex_count} vertices, {result.face_count} faces)"
        )
    ]
