from pathlib import Path

import typer

from splat.cli._console import console, error
from splat.domain.errors import SplatDomainError
from splat.handlers.tools.extract_surface import ExtractSurfaceRequest
from splat.registry.wiring import get_client


def extract_surface(
    input: Path = typer.Argument(...),
    output: Path = typer.Argument(...),
    to: str = typer.Option(None, "-t", "--to", help="obj | glb | gltf"),
    depth: int = typer.Option(
        9,
        "--depth",
        help="Poisson reconstruction octree depth.",
        envvar="SPLAT_EXTRACT_SURFACE_DEPTH",
    ),
    opacity_threshold: float = typer.Option(
        0.1,
        "--opacity-threshold",
        help="Drop Gaussians below this opacity before reconstruction.",
        envvar="SPLAT_EXTRACT_SURFACE_OPACITY_THRESHOLD",
    ),
) -> None:
    """Extract a textured surface mesh from a Gaussian splat (Poisson reconstruction)."""
    try:
        result = get_client().tools_extract_surface(
            ExtractSurfaceRequest(
                input_path=input,
                output_path=output,
                format=to,
                depth=depth,
                opacity_threshold=opacity_threshold,
            )
        )
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc

    console.print(
        f"[green]wrote[/green] {output} ({result.vertex_count:,} vertices, "
        f"{result.face_count:,} faces from {result.input_point_count:,} points)"
    )
