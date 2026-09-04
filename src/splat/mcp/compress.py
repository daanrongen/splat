from pathlib import Path

from splat.handlers.compress import CompressRequest
from splat.registry.wiring import get_client


def compress(input_path: str, output_path: str, profile: str = "web-delivery") -> str:
    """Prune outliers/low-opacity points and quantize for delivery."""
    cloud = get_client().compress(
        CompressRequest(input_path=Path(input_path), output_path=Path(output_path), profile=profile)
    )
    return f"wrote {output_path} ({cloud.point_count:,} points)"
