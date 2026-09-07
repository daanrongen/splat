from pathlib import Path

import typer

from splat.cli._console import console, error
from splat.domain.errors import SplatDomainError
from splat.handlers.tools.declutter import DeclutterRequest
from splat.registry.wiring import get_client


def declutter(
    input: Path = typer.Argument(...),
    output: Path = typer.Argument(...),
    k: int = typer.Option(
        16,
        "--k",
        help="Neighbors considered per point when estimating local density.",
        envvar="SPLAT_DECLUTTER_K",
    ),
    std_ratio: float = typer.Option(
        2.0,
        "--std-ratio",
        help="Points beyond this many std-devs of mean neighbor distance are removed.",
        envvar="SPLAT_DECLUTTER_STD_RATIO",
    ),
) -> None:
    """Remove isolated floater Gaussians via neighbor-density outlier detection."""
    try:
        cloud = get_client().tools_declutter(
            DeclutterRequest(input_path=input, output_path=output, k=k, std_ratio=std_ratio)
        )
    except (SplatDomainError, ValueError) as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc

    console.print(f"[green]wrote[/green] {output} ({cloud.point_count:,} points)")
