from pathlib import Path

import mcp.types as types

from splat.handlers.tools.declutter import DeclutterRequest
from splat.mcp._content import text_content
from splat.registry.wiring import get_client


def declutter(
    input_path: str,
    output_path: str,
    k: int = 16,
    std_ratio: float = 2.0,
) -> list[types.ContentBlock]:
    cloud = get_client().tools_declutter(
        DeclutterRequest(
            input_path=Path(input_path), output_path=Path(output_path), k=k, std_ratio=std_ratio
        )
    )
    return [text_content(f"wrote {output_path} ({cloud.point_count} points)")]
