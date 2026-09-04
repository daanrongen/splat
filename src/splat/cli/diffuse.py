from pathlib import Path

import typer

from splat.cli._console import console, error, warn
from splat.cli._pipeline_io import report
from splat.domain.errors import SplatDomainError
from splat.handlers.diffuse import DiffuseRequest, handle


def diffuse(
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
    """Diffuse an image from a text prompt (cached; the pipeline's origin stage)."""
    try:
        result = handle(
            DiffuseRequest(
                prompt=prompt,
                model=model,
                negative_prompt=negative_prompt,
                steps=steps,
                seed=seed,
                device=device,
            )
        )
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc

    if result.license_warning:
        warn(result.license_warning)

    if output is not None:
        output.write_bytes(result.asset.content_path.read_bytes())

    report(
        [result.asset],
        lambda assets: console.print(
            f"[green]diffused[/green] {assets[0].id} ({assets[0].content_path})"
        ),
    )
