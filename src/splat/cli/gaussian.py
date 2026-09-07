from pathlib import Path

import typer

from splat.cli._console import console, error, warn
from splat.cli._pipeline_io import is_piped, report, resolve_inputs
from splat.domain.errors import SplatDomainError
from splat.domain.manifest import Manifest, ManifestKind
from splat.handlers.gaussian import GaussianRequest
from splat.registry.wiring import get_client, get_manifest_repository, get_reader, get_writer


def _export_gaussian(asset: Manifest, output: Path) -> list[str]:
    cloud = get_reader(asset.content_path.suffix).read(asset.content_path)
    writer = get_writer(output.suffix)
    warnings = writer.supports(cloud)
    writer.write(cloud, output)
    return warnings


def gaussian(
    inputs: list[str] = typer.Argument(
        ..., help="Image paths, @<asset-id>s, or '-' to read piped asset records."
    ),
    output: Path | None = typer.Option(None, "-o", "--output", help="Output splat file path."),
    model: str = typer.Option(
        "mlx3d-capture",
        "--model",
        help="Reconstruction model, e.g. mlx3d-capture.",
        envvar="SPLAT_GAUSSIAN_MODEL",
    ),
    device: str = typer.Option(
        "auto", "--device", help="auto | cpu | mps", envvar="SPLAT_GAUSSIAN_DEVICE"
    ),
    quality: str = typer.Option(
        "fast", "--quality", help="mlx3d quality preset.", envvar="SPLAT_GAUSSIAN_QUALITY"
    ),
    iters: int | None = typer.Option(
        None, "--iters", help="Override training iterations.", envvar="SPLAT_GAUSSIAN_ITERS"
    ),
    max_dim: int | None = typer.Option(
        None, "--max-dim", help="Max training image dimension.", envvar="SPLAT_GAUSSIAN_MAX_DIM"
    ),
    sh_degree: int | None = typer.Option(
        None, "--sh-degree", help="Spherical harmonic degree.", envvar="SPLAT_GAUSSIAN_SH_DEGREE"
    ),
    poses: str = typer.Option(
        "auto",
        "--poses",
        help="auto | colmap | builtin | existing",
        envvar="SPLAT_GAUSSIAN_POSES",
    ),
    refine_poses: str = typer.Option(
        "auto", "--refine-poses", help="auto | on | off", envvar="SPLAT_GAUSSIAN_REFINE_POSES"
    ),
    low_memory: bool = typer.Option(
        False, "--low-mem", help="Use mlx3d low-memory mode.", envvar="SPLAT_GAUSSIAN_LOW_MEM"
    ),
    seed: int = typer.Option(
        0, "--seed", help="Random seed; <0 disables seeding.", envvar="SPLAT_GAUSSIAN_SEED"
    ),
) -> None:
    """Reconstruct a Gaussian splat from images."""
    cache = get_manifest_repository()
    try:
        assets: list[Manifest] = []
        for input_arg in inputs:
            assets.extend(resolve_inputs(input_arg, cache, default_kind=ManifestKind.IMAGE))
        results = get_client().gaussian(
            GaussianRequest(
                inputs=assets,
                model=model,
                device=device,
                quality=quality,
                iters=iters,
                max_dim=max_dim,
                sh_degree=sh_degree,
                poses=poses,
                refine_poses=refine_poses,
                low_memory=low_memory,
                seed=seed,
            )
        )
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc

    if output is not None and results:
        for warning in _export_gaussian(results[0], output):
            warn(warning)
        if not is_piped():
            console.print(
                f"[green]wrote[/green] {output} ({results[0].metadata.point_count:,} points)"
            )

    def _human(assets: list[Manifest]) -> None:
        for asset in assets:
            console.print(
                f"[green]gaussian[/green] {asset.id}  points={asset.metadata.point_count}"
            )

    report(results, _human)
