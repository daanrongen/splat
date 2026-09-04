from pathlib import Path

import typer

from splat.cli._console import console, error
from splat.cli._pipeline_io import report, resolve_inputs
from splat.domain.asset import AssetKind
from splat.domain.errors import SplatDomainError
from splat.handlers.mesh import MeshRequest, handle
from splat.registry.wiring import get_asset_cache


def mesh(
    input: str = typer.Argument(
        ..., help="Image/sticker path, @<asset-id>, or '-' to read piped asset records."
    ),
    output: Path | None = typer.Option(None, "-o", "--output", help="Write the mesh file here."),
    model: str = typer.Option("triposr", "--model", help="Mesh-prediction model."),
    device: str = typer.Option("auto", "--device", help="auto | cpu | mps"),
) -> None:
    """Predict a 3D mesh from an image using a learned model."""
    cache = get_asset_cache()
    try:
        inputs = resolve_inputs(input, cache, default_kind=AssetKind.IMAGE)
        results = handle(MeshRequest(inputs=inputs, model=model, device=device))
    except (SplatDomainError, NotImplementedError) as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc

    if output is not None and results:
        output.write_bytes(results[0].content_path.read_bytes())

    def _human(assets: list) -> None:
        for asset in assets:
            console.print(
                f"[green]mesh[/green] {asset.id}  faces={asset.metadata.get('face_count')}"
            )

    report(results, _human)
