from pathlib import Path

import typer

from splat.cli._console import console


def extract_surface(
    input: Path = typer.Argument(...),
    output: Path = typer.Argument(...),
    to: str = typer.Option(None, "-t", "--to", help="obj | gltf | usdz"),
) -> None:
    """Extract a surface mesh from a Gaussian splat. Not yet implemented."""
    console.print(
        "[yellow]not yet implemented[/yellow] - mesh export surface extraction is tracked for a "
        "future release; see ports/mesh.py's MeshExporter for the interface it will implement."
    )
    raise typer.Exit(code=1)
