from pathlib import Path

import typer
from rich.tree import Tree

from splat.cli import _table as table
from splat.cli._console import console, error
from splat.cli._pipeline_io import report, resolve_inputs
from splat.domain.errors import SplatDomainError
from splat.domain.manifest import Manifest
from splat.handlers import manifest as manifest_handler
from splat.registry.wiring import get_manifest_repository

manifest_app = typer.Typer(
    help="Inspect and manage cached pipeline manifests.", no_args_is_help=True
)


def _size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return ""


def _node(manifest: Manifest) -> str:
    from splat.application.models_admin import model_license

    label = f" [bold]{manifest.label}[/bold]" if manifest.label else ""
    license = model_license(manifest.created_by)
    line = f"{manifest.id}{label}  {manifest.kind.value}  {manifest.created_by}"
    line += f"  [dim]{license}[/dim]" if license else ""
    params = {k: v for k, v in manifest.params.items() if k != "device"}
    return line + (f"\n[dim]{params}[/dim]" if params else "")


def _tree(lineage: tuple[Manifest, list], tree: Tree | None = None) -> Tree:
    manifest, parents = lineage
    branch = Tree(_node(manifest)) if tree is None else tree.add(_node(manifest))
    for parent in parents:
        _tree(parent, branch)
    return branch


def _fail(exc: SplatDomainError) -> typer.Exit:
    error(str(exc))
    return typer.Exit(code=1)


@manifest_app.command("list")
def list_manifests(
    kind: str = typer.Option(None, help="Filter by kind, e.g. image, gaussian_cloud."),
    created_by: str = typer.Option(None, help="Filter by created_by substring, e.g. diffuse."),
    label: str = typer.Option(None, help="Filter by label substring."),
    limit: int = typer.Option(50, help="Max rows to show."),
) -> None:
    """List cached manifests, most recent first."""
    try:
        rows = manifest_handler.list_manifests(
            kind=kind, created_by=created_by, label=label, limit=limit
        )
    except SplatDomainError as exc:
        raise _fail(exc) from exc
    table.render(
        ("id", "kind", "label", "prompt", "size", "created by"),
        (
            (
                m.id,
                m.kind.value,
                m.label,
                str(m.params.get("prompt", "")),
                _size(m.content_size),
                m.created_by,
            )
            for m in rows
        ),
        flex=3,
    )


@manifest_app.command("get")
def get_manifest(manifest_id: str = typer.Argument(...)) -> None:
    """Show one manifest in full, with its lineage tree and derived manifests."""
    try:
        m = manifest_handler.get(manifest_id)
        lineage = manifest_handler.lineage(manifest_id)
        children = manifest_handler.children(manifest_id)
    except SplatDomainError as exc:
        raise _fail(exc) from exc
    console.print(f"id:           {m.id}")
    console.print(f"label:        {m.label}")
    console.print(f"kind:         {m.kind.value}")
    console.print(f"created_by:   {m.created_by}")
    console.print(f"created_at:   {m.created_at}")
    console.print(f"content_path: {m.content_path}")
    console.print(f"size:         {m.content_size:,} bytes")
    console.print(f"sha256:       {m.content_sha256}")
    console.print(f"params:       {m.params}")
    console.print(f"metadata:     {m.metadata}")
    console.print(f"children:     {[c.id for c in children]}")
    console.print("lineage:")
    console.print(_tree(lineage))


@manifest_app.command("label")
def label(
    input: str = typer.Argument(..., help="Manifest id/@id, or '-' to read piped asset records."),
    text: str = typer.Argument(..., help="Label to set; '' clears it."),
) -> None:
    """Name manifests. Piped records pass through, so it can sit mid-pipeline."""
    try:
        assets = resolve_inputs(
            input if input == "-" else f"@{input.removeprefix('@')}", get_manifest_repository()
        )
        labelled = [manifest_handler.label(asset.id, text) for asset in assets]
    except SplatDomainError as exc:
        raise _fail(exc) from exc
    report(
        labelled,
        lambda rows: [console.print(f"[green]labelled[/green] {m.id} {text!r}") for m in rows],
    )


@manifest_app.command("export")
def export(
    manifest_id: str = typer.Argument(...),
    out_dir: Path = typer.Argument(..., help="Directory to write the files and sidecars into."),
) -> None:
    """Copy a manifest and its cached ancestors, each with a .manifest.json sidecar."""
    try:
        written = manifest_handler.export(manifest_id, out_dir)
    except SplatDomainError as exc:
        raise _fail(exc) from exc
    for path in written:
        console.print(f"[green]exported[/green] {path}")


@manifest_app.command("rm")
def rm(
    manifest_id: str = typer.Argument(...),
    cascade: bool = typer.Option(
        False, "--cascade", help="Also delete every manifest derived from this one."
    ),
) -> None:
    """Delete a cached manifest; refuses if others derive from it unless --cascade."""
    try:
        manifest_handler.delete(manifest_id, cascade=cascade)
    except SplatDomainError as exc:
        raise _fail(exc) from exc
    console.print(f"[green]removed[/green] {manifest_id}")


@manifest_app.command("gc")
def gc() -> None:
    """Remove cache files no valid manifest owns (orphaned payloads, broken metadata)."""
    removed = manifest_handler.gc()
    for path in removed:
        console.print(f"[dim]removed {path.name}[/dim]")
    console.print(f"[green]gc[/green] removed {len(removed)} file(s)")


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
        raise _fail(exc) from exc

    if not targets:
        console.print("[dim]nothing to clear[/dim]")
        return

    if not yes:
        typer.confirm(
            f"Delete {len(targets)} cached manifest(s)? This cannot be undone.", abort=True
        )

    try:
        manifest_handler.clear(targets)
    except SplatDomainError as exc:
        raise _fail(exc) from exc
    console.print(f"[green]cleared[/green] {len(targets)} manifest(s)")
