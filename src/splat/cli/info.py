import json

import typer

from splat.cli._console import console, error
from splat.cli._input_path import resolve_input_path
from splat.domain.errors import SplatDomainError
from splat.registry.wiring import get_client


def info(
    path: str = typer.Argument(..., help="Splat file to inspect, or @<manifest-id>."),
    as_json: bool = typer.Option(False, "--json", help="Print the summary as JSON."),
) -> None:
    """Print point count, SH degree, bounding box, file size, and cloud metadata."""
    try:
        path = resolve_input_path(path)
        summary = get_client().info(path)
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc

    if as_json:
        print(json.dumps(summary.as_json()))
        return

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
    if summary.capture_camera_intrinsics is not None:
        fx, fy, cx, cy, width, height = summary.capture_camera_intrinsics
        console.print(
            f"intrinsics:   fx={fx:g} fy={fy:g} cx={cx:g} cy={cy:g} size={width:g}x{height:g}"
        )
