from pathlib import Path

import typer

from splat.application.inspect import ValidateUseCase
from splat.cli._console import console, error
from splat.domain.errors import SplatDomainError
from splat.registry.wiring import get_reader


def validate(
    path: Path = typer.Argument(..., help="Splat file to validate."),
    strict: bool = typer.Option(False, "--strict", help="Also flag valid-but-suspicious values."),
) -> None:
    """Check a splat file's domain invariants; exits non-zero on failure (CI-friendly)."""
    try:
        result = ValidateUseCase(get_reader(path.suffix)).execute(path, strict=strict)
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc

    if result.issues:
        for issue in result.issues:
            console.print(f"[yellow]issue:[/yellow] {issue}")
        raise typer.Exit(code=1)

    console.print(f"[green]valid[/green] — {result.cloud.point_count:,} points")
