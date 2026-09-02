from pathlib import Path

import typer

from splat.application.convert import ConvertUseCase
from splat.application.reconstruct import ReconstructUseCase
from splat.cli._console import console, error, warn
from splat.domain.errors import SplatDomainError
from splat.registry.wiring import (
    get_model_source,
    get_reader,
    get_reconstruction_backend,
    get_writer,
)


def convert(
    inputs: list[Path] = typer.Argument(
        ...,
        help="One splat file (INPUT OUTPUT positional form), or one-or-more images "
        "with -o/--output and --model for reconstruction.",
    ),
    output: Path | None = typer.Option(None, "-o", "--output", help="Output file path."),
    from_format: str | None = typer.Option(
        None,
        "-f",
        "--from",
        help="Force input format, e.g. .ply (inferred from extension by default).",
    ),
    to_format: str | None = typer.Option(
        None,
        "-t",
        "--to",
        help="Force output format, e.g. .splat (inferred from extension by default).",
    ),
    model: str | None = typer.Option(
        None, "--model", help="Reconstruction model to run on image inputs, e.g. mvsplat."
    ),
    device: str = typer.Option("auto", "--device", help="auto | cpu | mps"),
) -> None:
    """Convert between splat formats, or reconstruct a splat from images."""
    try:
        if model is not None:
            _reconstruct(inputs, output, model=model, device=device)
        else:
            _convert_format(inputs, output, from_format=from_format, to_format=to_format)
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc


def _convert_format(
    inputs: list[Path], output: Path | None, *, from_format: str | None, to_format: str | None
) -> None:
    if output is not None:
        if len(inputs) != 1:
            raise SplatDomainError("Provide exactly one input file when using -o/--output.")
        input_path, output_path = inputs[0], output
    else:
        if len(inputs) != 2:
            raise SplatDomainError(
                "Provide INPUT and OUTPUT paths, or use -o/--output "
                "(required for --model reconstruction)."
            )
        input_path, output_path = inputs

    reader = get_reader(from_format or input_path.suffix)
    writer = get_writer(to_format or output_path.suffix)
    result = ConvertUseCase(reader, writer).execute(input_path, output_path)
    for warning in result.warnings:
        warn(warning)
    console.print(f"[green]wrote[/green] {output_path} ({result.cloud.point_count:,} points)")


def _reconstruct(inputs: list[Path], output: Path | None, *, model: str, device: str) -> None:
    if output is None:
        raise SplatDomainError("Reconstruction requires -o/--output.")

    model_source = get_model_source()
    backend = get_reconstruction_backend(model, model_source=model_source, device=device)

    min_images, max_images = backend.required_image_count()
    if len(inputs) < min_images or (max_images is not None and len(inputs) > max_images):
        upper = max_images if max_images is not None else "∞"
        raise SplatDomainError(
            f"Model {model!r} requires between {min_images} and {upper} images, got {len(inputs)}."
        )

    cloud = ReconstructUseCase(backend).execute(inputs, device=device)
    writer = get_writer(output.suffix)
    for warning in writer.supports(cloud):
        warn(warning)
    writer.write(cloud, output)
    console.print(f"[green]wrote[/green] {output} ({cloud.point_count:,} points)")
