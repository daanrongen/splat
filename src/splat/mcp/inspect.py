from pathlib import Path

from splat.handlers.inspect import info as info_handler
from splat.handlers.inspect import validate as validate_handler


def info(path: str) -> dict:
    """Print point count, SH degree, and bounding box for a splat file."""
    cloud = info_handler(Path(path))
    return {
        "format": cloud.metadata.source_format,
        "points": cloud.point_count,
        "sh_degree": cloud.sh_degree,
        "bbox_min": cloud.means.min(axis=0).tolist(),
        "bbox_max": cloud.means.max(axis=0).tolist(),
    }


def validate(path: str, strict: bool = False) -> dict:
    """Check a splat file's domain invariants."""
    result = validate_handler(Path(path), strict=strict)
    return {"valid": not result.issues, "issues": result.issues, "points": result.cloud.point_count}
