from pathlib import Path

import mcp.types as types

from splat.handlers.tools.compress import CompressRequest
from splat.mcp._content import text_content
from splat.registry.wiring import get_client


def compress(
    input_path: str, output_path: str, profile: str = "web-delivery"
) -> list[types.ContentBlock]:
    cloud = get_client().tools_compress(
        CompressRequest(input_path=Path(input_path), output_path=Path(output_path), profile=profile)
    )
    return [text_content(f"wrote {output_path} ({cloud.point_count} points)")]
