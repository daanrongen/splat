from pathlib import Path

import typer

from splat.adapters.compression.prune_quantize import PROFILES
from splat.cli._console import console, error
from splat.domain.errors import SplatDomainError
from splat.handlers.compress import CompressRequest, handle


def compress(
    input: Path = typer.Argument(...),
    output: Path = typer.Argument(...),
    profile: str = typer.Option("web-delivery", "--profile", help=f"One of: {', '.join(PROFILES)}"),
) -> None:
    """Prune outliers/low-opacity points and quantize for delivery."""
    try:
        cloud = handle(CompressRequest(input_path=input, output_path=output, profile=profile))
    except (SplatDomainError, ValueError) as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc

    console.print(f"[green]wrote[/green] {output} ({cloud.point_count:,} points)")
