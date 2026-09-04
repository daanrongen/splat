from pathlib import Path

import mcp.types as types

from splat.handlers.tools.convert import ConvertRequest
from splat.mcp._content import text_content
from splat.registry.wiring import get_client


def convert(
    input_path: str,
    output_path: str,
    from_format: str | None = None,
    to_format: str | None = None,
) -> list[types.ContentBlock]:
    result = get_client().convert(
        ConvertRequest(
            input_path=Path(input_path),
            output_path=Path(output_path),
            from_format=from_format,
            to_format=to_format,
        )
    )
    return [text_content(f"wrote {output_path} ({result.cloud.point_count} points)")]
