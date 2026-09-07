import typer

from splat.cli._console import console, error
from splat.cli._pipeline_io import report, resolve_inputs
from splat.domain.errors import SplatDomainError
from splat.domain.manifest import Manifest, ManifestKind
from splat.handlers.tools.normalize_color import NormalizeColorRequest, handle
from splat.registry.wiring import get_manifest_repository


def normalize_color(
    inputs: list[str] = typer.Argument(
        ..., help="Image/sticker paths or @<asset-id>s captured together (2+ recommended)."
    ),
) -> None:
    """Correct per-view exposure/white-balance drift across a multi-photo capture."""
    cache = get_manifest_repository()
    try:
        assets: list[Manifest] = []
        for input_arg in inputs:
            assets.extend(resolve_inputs(input_arg, cache, default_kind=ManifestKind.IMAGE))
        results = handle(NormalizeColorRequest(inputs=assets))
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc

    def _human(assets: list[Manifest]) -> None:
        console.print(f"[green]normalize.color[/green] corrected {len(assets)} images")
        for asset in assets:
            console.print(f"  {asset.id}")

    report(results, _human)
