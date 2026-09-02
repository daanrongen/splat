from pathlib import Path

import typer

from splat.cli._console import console


def export(
    input: Path = typer.Argument(...),
    output: Path = typer.Argument(...),
    to: str = typer.Option(None, "-t", "--to", help="obj | gltf | usdz"),
) -> None:
    """Export a splat to a mesh format. Not yet implemented."""
    console.print(
        "[yellow]not yet implemented[/yellow] — mesh export (SuGaR-style surface extraction) is "
        "tracked for a future release; see ports/mesh.py for the interface it will implement."
    )
    raise typer.Exit(code=1)
