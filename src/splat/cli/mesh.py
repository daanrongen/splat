from pathlib import Path

import numpy as np
import typer

from splat.adapters.cache.filesystem import FilesystemAssetCache
from splat.application.pipeline import run_depth, run_mesh
from splat.cli._console import console, error
from splat.cli._pipeline_io import report, resolve_inputs
from splat.domain.asset import Asset, AssetKind
from splat.domain.errors import SplatDomainError
from splat.domain.image_space import DepthMap
from splat.registry.wiring import get_depth_backend, get_mesh_backend, is_known_format


def mesh(
    input: str = typer.Argument(
        ..., help="Image/splat path, @<asset-id>, or '-' to read piped asset records."
    ),
    output: Path | None = typer.Option(None, "-o", "--output", help="Write the mesh file here."),
    model: str = typer.Option("depth-heightfield", "--model", help="Mesh-prediction model."),
    to: str = typer.Option("glb", "-t", "--to", help="Mesh export format: glb | obj | ply."),
    depth_model: str = typer.Option(
        "depth-pro", "--depth-model", help="Depth model run when a bare image has none yet."
    ),
    device: str = typer.Option("auto", "--device", help="auto | cpu | mps"),
) -> None:
    """Predict a 3D mesh from an image(+depth) — or export a Gaussian splat to
    a mesh file (not yet implemented)."""
    if is_known_format(Path(input).suffix):
        console.print(
            "[yellow]not yet implemented[/yellow] — gaussian -> mesh export (SuGaR-style "
            "surface extraction) is tracked for a future release; see registry/mesh.py's "
            "MESH_CATALOG and ports/mesh.py's MeshExporter for the interface it will implement."
        )
        raise typer.Exit(code=1)

    cache = FilesystemAssetCache()
    try:
        backend = get_mesh_backend(model, device=device)
        inputs = resolve_inputs(input, cache, default_kind=AssetKind.IMAGE)
        results = [
            _predict_one(asset, cache, backend, model, depth_model, to, device) for asset in inputs
        ]
    except SplatDomainError as exc:
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


def _predict_one(
    asset: Asset, cache, backend, model_name: str, depth_model: str, export_format: str, device: str
) -> Asset:
    if asset.kind == AssetKind.DEPTH_MAP:
        if not asset.parent_ids:
            raise SplatDomainError(
                f"Depth asset {asset.id} has no source image to texture the mesh with."
            )
        image_asset = cache.get(asset.parent_ids[0])
        depth_asset = asset
    else:
        image_asset = asset
        depth_backend = get_depth_backend(depth_model, device=device)
        depth_asset = run_depth(
            depth_backend, cache, model_name=depth_model, input_asset=image_asset, params={}
        )

    depth_map = _load_depth_map(depth_asset)
    return run_mesh(
        backend,
        cache,
        model_name=model_name,
        image_asset=image_asset,
        depth_asset=depth_asset,
        depth_map=depth_map,
        params={},
        export_format=export_format,
    )


def _load_depth_map(depth_asset: Asset) -> DepthMap:
    depth = np.load(depth_asset.content_path)
    return DepthMap(
        depth=depth,
        focal_length_px=depth_asset.metadata.get("focal_length_px"),
        field_of_view_deg=depth_asset.metadata.get("field_of_view_deg"),
        metadata=depth_asset.metadata,
    )
