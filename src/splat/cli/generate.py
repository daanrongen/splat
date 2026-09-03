from pathlib import Path

import typer

from splat.adapters.cache.filesystem import FilesystemAssetCache
from splat.application.pipeline import run_generate
from splat.cli._console import console, error, warn
from splat.cli._pipeline_io import report
from splat.domain.errors import SplatDomainError
from splat.registry.wiring import get_generation_backend


def generate(
    prompt: str = typer.Argument(..., help="Text prompt to generate an image from."),
    output: Path | None = typer.Option(None, "-o", "--output", help="Also export to this path."),
    model: str = typer.Option("sdxl-turbo-mlx", "--model"),
    negative_prompt: str = typer.Option("", "--negative"),
    steps: int | None = typer.Option(
        None, "--steps", help="Denoising steps (model-dependent default)."
    ),
    seed: int | None = typer.Option(None, "--seed"),
    device: str = typer.Option("auto", "--device", help="auto | cpu | mps"),
) -> None:
    """Generate an image from a text prompt (cached; the pipeline's origin stage)."""
    cache = FilesystemAssetCache()
    try:
        backend = get_generation_backend(model, device=device)
        if not backend.license.is_commercial:
            warn(f"{model} license: {backend.license}")
        asset = run_generate(
            backend,
            cache,
            model_name=model,
            prompt=prompt,
            params={"negative_prompt": negative_prompt, "steps": steps, "seed": seed},
        )
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc

    if output is not None:
        output.write_bytes(asset.content_path.read_bytes())

    report(
        [asset],
        lambda assets: console.print(
            f"[green]generated[/green] {assets[0].id} ({assets[0].content_path})"
        ),
    )
