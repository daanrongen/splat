from pathlib import Path

import mcp.types as types

from splat.handlers.convert import ConvertRequest
from splat.mcp._content import text_content
from splat.registry.wiring import get_client


def convert(
    input_path: str,
    output_path: str,
    from_format: str | None = None,
    to_format: str | None = None,
) -> list[types.ContentBlock]:
    """Convert between splat file formats."""
    result = get_client().convert(
        ConvertRequest(
            input_path=Path(input_path),
            output_path=Path(output_path),
            from_format=from_format,
            to_format=to_format,
        )
    )
    content = [text_content(f"wrote {output_path} ({result.cloud.point_count:,} points)")]
    content.extend(text_content(f"warning: {w}") for w in result.warnings)
    return content
