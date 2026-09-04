from pathlib import Path

import typer

from splat.cli._console import console, error
from splat.domain.errors import SplatDomainError
from splat.handlers.inspect import info as info_handler


def info(path: Path = typer.Argument(..., help="Splat file to inspect.")) -> None:
    """Print point count, SH degree, bounding box, and file size."""
    try:
        cloud = info_handler(path)
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc

    bbox_min = cloud.means.min(axis=0)
    bbox_max = cloud.means.max(axis=0)
    console.print(f"format:       {cloud.metadata.source_format}")
    console.print(f"points:       {cloud.point_count:,}")
    console.print(f"sh degree:    {cloud.sh_degree}")
    console.print(f"bounding box: {bbox_min.tolist()} .. {bbox_max.tolist()}")
    console.print(f"file size:    {path.stat().st_size:,} bytes")
