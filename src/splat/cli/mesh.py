from pathlib import Path

import typer

from splat.cli._console import console, error
from splat.cli._pipeline_io import export_output, prepare_output, report, resolve_inputs
from splat.domain.errors import SplatDomainError
from splat.domain.manifest import ManifestKind
from splat.handlers.mesh import MeshRequest, handle
from splat.registry.wiring import get_manifest_repository


def mesh(
    input: str = typer.Argument(
        ..., help="Depth map or Gaussian cloud: path/@<asset-id>, or '-' for piped records."
    ),
    output: Path | None = typer.Option(None, "-o", "--output", help="Also write the mesh here."),
    model: str | None = typer.Option(
        None,
        "--model",
        help="heightfield (depth map) | poisson (gaussian cloud); default follows the input.",
        envvar="SPLAT_MESH_MODEL",
    ),
    to: str | None = typer.Option(
        None,
        "--to",
        help="glb | obj | ply | gltf; default from -o, else glb.",
        envvar="SPLAT_MESH_TO",
    ),
    depth: int = typer.Option(
        9, "--depth", help="poisson: octree depth.", envvar="SPLAT_MESH_DEPTH"
    ),
    opacity_threshold: float = typer.Option(
        0.1,
        "--opacity-threshold",
        help="poisson: drop Gaussians below this opacity first.",
        envvar="SPLAT_MESH_OPACITY_THRESHOLD",
    ),
) -> None:
    """Turn a metric depth map or a Gaussian cloud into a textured mesh."""
    fmt = to or (output.suffix.lstrip(".") if output else "glb")
    cache = get_manifest_repository()
    try:
        prepare_output(output)
        inputs = resolve_inputs(input, cache, default_kind=ManifestKind.GAUSSIAN_CLOUD)
        results = handle(
            MeshRequest(
                inputs=inputs,
                model=model,
                format=fmt,
                depth=depth,
                opacity_threshold=opacity_threshold,
            )
        )
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc

    if output is not None and results:
        export_output(results[0], output, cache)

    def _human(assets: list) -> None:
        for asset in assets:
            extra = asset.metadata.extra
            console.print(
                f"[green]mesh[/green] {asset.id}  {extra.get('vertex_count', 0):,} vertices  "
                f"{extra.get('face_count', 0):,} faces"
            )

    report(results, _human)
