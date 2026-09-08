from pathlib import Path

import typer

from splat.cli._console import console, error
from splat.domain.errors import SplatDomainError
from splat.registry.wiring import get_client


def info(path: Path = typer.Argument(..., help="Splat file to inspect.")) -> None:
    """Print point count, SH degree, bounding box, file size, and cloud metadata."""
    try:
        summary = get_client().info(path)
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc

    console.print(f"format:       {summary.format}")
    console.print(f"points:       {summary.points:,}")
    console.print(f"sh degree:    {summary.sh_degree}")
    console.print(f"bounding box: {summary.bbox_min} .. {summary.bbox_max}")
    console.print(f"file size:    {path.stat().st_size:,} bytes")
    console.print(f"convention:   {summary.coordinate_convention} (up={summary.up_axis})")
    if summary.source_model is not None:
        console.print(f"source model: {summary.source_model}")
    if summary.license is not None:
        console.print(f"license:      {summary.license}")
    if summary.capture_camera_count is not None:
        console.print(f"cameras:      {summary.capture_camera_count}")
