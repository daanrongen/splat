from pathlib import Path

import typer

from splat.cli._console import console, error, warn
from splat.domain.errors import SplatDomainError
from splat.handlers.convert import ConvertRequest
from splat.registry.wiring import get_client


def convert(
    inputs: list[Path] = typer.Argument(
        ..., help="INPUT OUTPUT positional form, or one INPUT with -o/--output."
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
) -> None:
    """Convert between splat file formats."""
    try:
        if output is not None:
            if len(inputs) != 1:
                raise SplatDomainError("Provide exactly one input file when using -o/--output.")
            input_path, output_path = inputs[0], output
        else:
            if len(inputs) != 2:
                raise SplatDomainError("Provide INPUT and OUTPUT paths, or use -o/--output.")
            input_path, output_path = inputs

        result = get_client().convert(
            ConvertRequest(
                input_path=input_path,
                output_path=output_path,
                from_format=from_format,
                to_format=to_format,
            )
        )
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc

    for warning in result.warnings:
        warn(warning)
    console.print(f"[green]wrote[/green] {output_path} ({result.cloud.point_count:,} points)")
