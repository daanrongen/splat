from pathlib import Path

import typer

from splat.cli import _table as table
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
    # created_by is the widest, least predictable column ("gaussian:mlx3d-capture"
    # vs "external"), so it absorbs the leftover width instead of the id/kind/
    # timestamp columns, which are all fixed-width already.
    table.render(
        ("id", "kind", "created by", "created at"),
        ((m.id, m.kind.value, m.created_by, m.created_at) for m in rows),
        flex=2,
    )


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


@manifest_app.command("export")
def export(
    manifest_id: str = typer.Argument(...),
    out_dir: Path = typer.Argument(..., help="Directory to write the files and sidecars into."),
) -> None:
    """Copy a manifest and its cached ancestors, each with a .manifest.json sidecar."""
    try:
        written = manifest_handler.export(manifest_id, out_dir)
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc
    for path in written:
        console.print(f"[green]exported[/green] {path}")


@manifest_app.command("rm")
def rm(manifest_id: str = typer.Argument(...)) -> None:
    """Delete a cached manifest's content and metadata."""
    try:
        manifest_handler.delete(manifest_id)
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc
    console.print(f"[green]removed[/green] {manifest_id}")


@manifest_app.command("clear")
def clear(
    kind: str = typer.Option(None, help="Only clear manifests of this kind, e.g. image."),
    created_by: str = typer.Option(
        None, help="Only clear manifests whose created_by contains this."
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the confirmation prompt and clear immediately."
    ),
) -> None:
    """Delete every cached manifest, or just those matching --kind/--created-by."""
    try:
        targets = manifest_handler.list_manifests(kind=kind, created_by=created_by)
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc

    if not targets:
        console.print("[dim]nothing to clear[/dim]")
        return

    if not yes:
        typer.confirm(
            f"Delete {len(targets)} cached manifest(s)? This cannot be undone.", abort=True
        )

    for target in targets:
        manifest_handler.delete(target.id)
    console.print(f"[green]cleared[/green] {len(targets)} manifest(s)")
