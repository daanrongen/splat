from pathlib import Path

import typer

from splat.application.reconstruct import ReconstructUseCase
from splat.cli._console import console, error, warn
from splat.domain.errors import SplatDomainError
from splat.registry.wiring import get_model_source, get_reconstruction_backend, get_writer


def gaussian(
    inputs: list[Path] = typer.Argument(
        ..., help="Two or more images to reconstruct into a Gaussian splat."
    ),
    output: Path = typer.Option(..., "-o", "--output", help="Output splat file path."),
    model: str = typer.Option("mvsplat", "--model", help="Reconstruction model, e.g. mvsplat."),
    device: str = typer.Option("auto", "--device", help="auto | cpu | mps"),
) -> None:
    """Reconstruct a Gaussian splat from images (feed-forward, no per-scene optimization)."""
    try:
        model_source = get_model_source()
        backend = get_reconstruction_backend(model, model_source=model_source, device=device)

        min_images, max_images = backend.required_image_count()
        if len(inputs) < min_images or (max_images is not None and len(inputs) > max_images):
            upper = max_images if max_images is not None else "∞"
            raise SplatDomainError(
                f"Model {model!r} requires between {min_images} and {upper} images, "
                f"got {len(inputs)}."
            )

        cloud = ReconstructUseCase(backend).execute(inputs, device=device)
        writer = get_writer(output.suffix)
        for warning in writer.supports(cloud):
            warn(warning)
        writer.write(cloud, output)
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc

    console.print(f"[green]wrote[/green] {output} ({cloud.point_count:,} points)")
