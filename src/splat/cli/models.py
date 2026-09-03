import typer

from splat.application.models_admin import (
    ListModelsUseCase,
    ModelInfoUseCase,
    PullModelUseCase,
    RemoveModelUseCase,
)
from splat.cli._console import console, error
from splat.domain.errors import SplatDomainError
from splat.registry.wiring import get_model_source

models_app = typer.Typer(
    help="Manage locally cached models (reconstruction, generation, segmentation, depth).",
    no_args_is_help=True,
)


@models_app.command("list")
def list_models() -> None:
    """List catalog models and whether their weights are cached locally."""
    rows = ListModelsUseCase(get_model_source()).execute()
    for descriptor, cached in rows:
        status = "cached" if cached else "not pulled"
        runtime = getattr(descriptor, "runtime", "-")
        console.print(f"{descriptor.name:20} {runtime:8} {descriptor.license!s:24} {status}")


@models_app.command("pull")
def pull(name: str = typer.Argument(...)) -> None:
    """Download a model's weights from HuggingFace Hub."""
    try:
        PullModelUseCase(get_model_source()).execute(name)
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc
    console.print(f"[green]pulled[/green] {name}")


@models_app.command("info")
def info(name: str = typer.Argument(...)) -> None:
    """Show a model's license, source repo, and expected input shape."""
    try:
        descriptor = ModelInfoUseCase().execute(name)
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc
    console.print(f"name:    {descriptor.name}")
    console.print(f"runtime: {getattr(descriptor, 'runtime', '-')}")
    console.print(f"source:  {descriptor.hf_repo_id}")
    console.print(f"license: {descriptor.license}")
    if hasattr(descriptor, "min_images"):
        upper = descriptor.max_images if descriptor.max_images is not None else "∞"
        console.print(f"images:  {descriptor.min_images}..{upper}")


@models_app.command("rm")
def rm(name: str = typer.Argument(...)) -> None:
    """Remove a model's cached weights."""
    try:
        RemoveModelUseCase(get_model_source()).execute(name)
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc
    console.print(f"[green]removed[/green] {name}")
