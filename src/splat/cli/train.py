from pathlib import Path

import typer

from splat.cli._console import console


def train(
    dataset_dir: Path = typer.Argument(...),
) -> None:
    """Per-scene optimization training. Not yet implemented."""
    _ = dataset_dir
    console.print(
        "[yellow]not yet implemented[/yellow] - per-scene Gaussian splat optimization "
        "is intentionally deferred until a concrete backend is selected."
    )
    raise typer.Exit(code=1)
