import typer

from splat.cli._console import console, error
from splat.domain.errors import SplatDomainError
from splat.registry.wiring import get_client

models_app = typer.Typer(help="Manage locally cached model weights.", no_args_is_help=True)


@models_app.command("list")
def list_models() -> None:
    """List catalog models and whether their weights are cached locally."""
    rows = get_client().models_list()
    for row in rows:
        status = "cached" if row.cached else "not pulled"
        console.print(f"{row.name:20} {row.runtime:8} {row.license:24} {status}")


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


@models_app.command("rm")
def rm(name: str = typer.Argument(...)) -> None:
    """Remove a model's cached weights."""
    try:
        get_client().models_rm(name)
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc
    console.print(f"[green]removed[/green] {name}")
