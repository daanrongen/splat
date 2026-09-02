from pathlib import Path

import typer

from splat.cli._console import console


def train(
    dataset_dir: Path = typer.Argument(...),
    iterations: int = typer.Option(30_000, "--iterations"),
    backend: str = typer.Option(None, "--backend"),
) -> None:
    """Per-scene optimization training. Not yet implemented."""
    console.print(
        "[yellow]not yet implemented[/yellow] — per-scene optimization training is deferred: "
        "the reference rasterizer is CUDA-only with no settled Apple Silicon equivalent yet. "
        "Candidates being tracked: gsplat-mlx, msplat, OpenSplat (MPS via libtorch). "
        "See ports/training.py."
    )
    raise typer.Exit(code=1)
