from pathlib import Path

import typer

from splat.cli._console import console, error, warn
from splat.domain.errors import SplatDomainError
from splat.handlers.gaussian import GaussianRequest
from splat.registry.wiring import get_client


def gaussian(
    inputs: list[Path] = typer.Argument(
        ..., help="Two or more images to reconstruct into a Gaussian splat."
    ),
    output: Path = typer.Option(..., "-o", "--output", help="Output splat file path."),
    model: str = typer.Option("mvsplat", "--model", help="Reconstruction model, e.g. mvsplat."),
    device: str = typer.Option("auto", "--device", help="auto | cpu | mps"),
) -> None:
    """Reconstruct a Gaussian splat from images (feed-forward, no per-scene optimization)."""
    try:
        result = get_client().gaussian(
            GaussianRequest(inputs=inputs, output_path=output, model=model, device=device)
        )
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc

    for warning in result.warnings:
        warn(warning)
    console.print(f"[green]wrote[/green] {output} ({result.cloud.point_count:,} points)")
