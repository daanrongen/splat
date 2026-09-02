from pathlib import Path

import typer

from splat.adapters.compression.prune_quantize import PROFILES, PruneQuantizeCompressor
from splat.application.compress import CompressUseCase
from splat.cli._console import console, error
from splat.domain.errors import SplatDomainError
from splat.registry.wiring import get_reader, get_writer


def compress(
    input: Path = typer.Argument(...),
    output: Path = typer.Argument(...),
    profile: str = typer.Option("web-delivery", "--profile", help=f"One of: {', '.join(PROFILES)}"),
) -> None:
    """Prune outliers/low-opacity points and quantize for delivery."""
    try:
        reader = get_reader(input.suffix)
        writer = get_writer(output.suffix)
        result = CompressUseCase(reader, writer, PruneQuantizeCompressor()).execute(
            input, output, profile=profile
        )
    except (SplatDomainError, ValueError) as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc

    console.print(f"[green]wrote[/green] {output} ({result.point_count:,} points)")
