from pathlib import Path

import typer

from splat.adapters.compression.prune_quantize import PROFILES
from splat.cli._console import console, error
from splat.cli._input_path import resolve_input_path
from splat.domain.errors import SplatDomainError
from splat.handlers.tools.compress import CompressRequest
from splat.registry.wiring import get_client


def compress(
    input: str = typer.Argument(..., help="Splat file to compress, or @<manifest-id>."),
    output: Path = typer.Argument(...),
    profile: str = typer.Option(
        "web-delivery",
        "--profile",
        help=f"One of: {', '.join(PROFILES)}",
        envvar="SPLAT_COMPRESS_PROFILE",
    ),
    pruning: str = typer.Option(
        "threshold",
        "--pruning",
        help="threshold | blue-noise (spatially-uniform, needs --target-count)",
        envvar="SPLAT_COMPRESS_PRUNING",
    ),
    target_count: int | None = typer.Option(
        None, "--target-count", help="Target point count for --pruning blue-noise."
    ),
) -> None:
    """Prune outliers/low-opacity points and quantize for delivery."""
    try:
        cloud = get_client().tools_compress(
            CompressRequest(
                input_path=resolve_input_path(input),
                output_path=output,
                profile=profile,
                pruning=pruning,
                target_count=target_count,
            )
        )
    except (SplatDomainError, ValueError) as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc

    console.print(f"[green]wrote[/green] {output} ({cloud.point_count:,} points)")
