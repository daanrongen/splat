from pathlib import Path

from splat.registry.wiring import get_client


def info(path: str) -> dict:
    """Print point count, SH degree, and bounding box for a splat file."""
    summary = get_client().info(Path(path))
    return {
        "format": summary.format,
        "points": summary.points,
        "sh_degree": summary.sh_degree,
        "bbox_min": summary.bbox_min,
        "bbox_max": summary.bbox_max,
    }


def validate(path: str, strict: bool = False) -> dict:
    """Check a splat file's domain invariants."""
    summary = get_client().validate(Path(path), strict=strict)
    return {"valid": summary.valid, "issues": summary.issues, "points": summary.points}
