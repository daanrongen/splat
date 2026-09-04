from pathlib import Path

import typer

from splat.cli._console import console, error
from splat.cli._pipeline_io import report, resolve_inputs
from splat.domain.asset import AssetKind
from splat.domain.errors import SplatDomainError
from splat.handlers.tools.displace_height import DisplaceHeightRequest, handle
from splat.registry.wiring import get_asset_cache


def displace_height(
    input: str = typer.Argument(
        ..., help="Depth-map path, @<asset-id>, or '-' - its source image is fetched by provenance."
    ),
    output: Path | None = typer.Option(None, "-o", "--output", help="Write the mesh file here."),
    to: str | None = typer.Option(
        None, "-t", "--to", help="Mesh format: glb | obj | ply (inferred from -o by default)."
    ),
) -> None:
    """Displace a depth map's per-pixel height into a triangulated, textured mesh."""
    export_format = to or (output.suffix.lstrip(".") if output else "glb")
    cache = get_asset_cache()
    try:
        inputs = resolve_inputs(input, cache, default_kind=AssetKind.DEPTH_MAP)
        results = handle(DisplaceHeightRequest(inputs=inputs, export_format=export_format))
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc

    if output is not None and results:
        output.write_bytes(results[0].content_path.read_bytes())

    def _human(assets: list) -> None:
        for asset in assets:
            faces = asset.metadata.get("face_count")
            console.print(f"[green]displace.height[/green] {asset.id}  faces={faces}")

    report(results, _human)
