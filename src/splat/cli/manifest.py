import typer

from splat.cli._console import console, error
from splat.domain.errors import SplatDomainError
from splat.handlers import manifest as manifest_handler

manifest_app = typer.Typer(
    help="Inspect and manage cached pipeline manifests.", no_args_is_help=True
)


@manifest_app.command("list")
def list_manifests(
    kind: str = typer.Option(None, help="Filter by kind, e.g. image, gaussian_cloud."),
    created_by: str = typer.Option(None, help="Filter by created_by substring, e.g. diffuse."),
    limit: int = typer.Option(50, help="Max rows to show."),
) -> None:
    """List cached manifests, most recent first."""
    try:
        rows = manifest_handler.list_manifests(kind=kind, created_by=created_by, limit=limit)
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc
    for m in rows:
        console.print(f"{m.id:18} {m.kind.value:14} {m.created_by:24} {m.created_at}")


@manifest_app.command("get")
def get_manifest(manifest_id: str = typer.Argument(...)) -> None:
    """Show full detail for one cached manifest."""
    try:
        m = manifest_handler.get(manifest_id)
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc
    console.print(f"id:           {m.id}")
    console.print(f"kind:         {m.kind.value}")
    console.print(f"created_by:   {m.created_by}")
    console.print(f"created_at:   {m.created_at}")
    console.print(f"content_path: {m.content_path}")
    console.print(f"size:         {m.content_size:,} bytes")
    console.print(f"sha256:       {m.content_sha256}")
    console.print(f"parent_ids:   {m.parent_ids}")
    console.print(f"params:       {m.params}")
    console.print(f"metadata:     {m.metadata}")


@manifest_app.command("rm")
def rm(manifest_id: str = typer.Argument(...)) -> None:
    """Delete a cached manifest's content and metadata."""
    try:
        manifest_handler.delete(manifest_id)
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc
    console.print(f"[green]removed[/green] {manifest_id}")
