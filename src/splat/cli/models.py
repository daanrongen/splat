import typer

from splat.cli import _table as table
from splat.cli._console import console, error, warn
from splat.domain.errors import SplatDomainError
from splat.registry.wiring import get_client

models_app = typer.Typer(help="Manage locally cached model weights.", no_args_is_help=True)


@models_app.command("list")
def list_models() -> None:
    """List catalog models, the stage each serves, and whether they're cached."""
    rows = get_client().models_list()
    # license last: it is the widest and least scanned column, so it wraps at
    # the right edge instead of pushing the names around.
    table.render(
        ("model", "for", "runtime", "on disk", "license"),
        (
            (
                row.name,
                row.stage,
                row.runtime,
                "[green]yes[/green]" if row.cached else "[dim]no[/dim]",
                row.license if row.commercial else f"[red]{row.license}[/red]",
            )
            for row in rows
        ),
        flex=4,
        caption="red license = non-commercial / research use only",
    )


@models_app.command("pull")
def pull(name: str = typer.Argument(...)) -> None:
    """Download a model's weights from HuggingFace Hub."""
    try:
        get_client().models_pull(name)
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc
    console.print(f"[green]pulled[/green] {name}")


@models_app.command("info")
def info(name: str = typer.Argument(...)) -> None:
    """Show a model's license, source repo, and expected input shape."""
    try:
        info = get_client().models_info(name)
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc
    console.print(f"name:    {info.name}")
    console.print(f"runtime: {info.runtime}")
    console.print(f"source:  {info.source}")
    console.print(f"license: {info.license}")
    if info.min_images is not None:
        upper = info.max_images if info.max_images is not None else "∞"
        console.print(f"images:  {info.min_images}..{upper}")
    if info.dimension is not None:
        console.print(f"dim:     {info.dimension}")
    if info.normalized is not None:
        console.print(f"norm:    {info.normalized}")
    if info.notes:
        console.print(f"notes:   {info.notes}")


def _human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:,.1f} {unit}"
        value /= 1024
    return f"{value:,.1f} GB"


@models_app.command("prune")
def prune(
    yes: bool = typer.Option(False, "--yes", help="Actually delete; otherwise just report."),
) -> None:
    """Report (or with --yes, delete) cached weights no catalog model uses."""
    from splat.handlers import models as models_handler

    orphans = models_handler.prune(apply=yes)
    if not orphans:
        console.print("[green]nothing to prune[/green]")
        return

    for orphan in orphans:
        verb = "removed" if yes else "orphaned"
        color = "green" if yes else "yellow"
        console.print(
            f"[{color}]{verb}[/{color}] {orphan.repo_id:48} {_human_bytes(orphan.size_bytes)}"
        )

    total = _human_bytes(sum(orphan.size_bytes for orphan in orphans))
    if yes:
        console.print(f"reclaimed {total}")
    else:
        warn(f"{total} reclaimable; re-run with --yes to delete")


@models_app.command("rm")
def rm(name: str = typer.Argument(...)) -> None:
    """Remove a model's cached weights."""
    try:
        get_client().models_rm(name)
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc
    console.print(f"[green]removed[/green] {name}")
