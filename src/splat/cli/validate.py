import typer

from splat.cli._console import console, error
from splat.cli._input_path import resolve_input_path
from splat.domain.errors import SplatDomainError
from splat.registry.wiring import get_client


def validate(
    path: str = typer.Argument(..., help="Splat file to validate, or @<manifest-id>."),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Also flag valid-but-suspicious values.",
        envvar="SPLAT_VALIDATE_STRICT",
    ),
) -> None:
    """Check a splat file's domain invariants; exits non-zero on failure (CI-friendly)."""
    try:
        path = resolve_input_path(path)
        summary = get_client().validate(path, strict=strict)
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc

    if summary.issues:
        for issue in summary.issues:
            console.print(f"[yellow]issue:[/yellow] {issue}")
        raise typer.Exit(code=1)

    console.print(f"[green]valid[/green] — {summary.points:,} points")
