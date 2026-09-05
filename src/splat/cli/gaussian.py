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
        "mlx3d-capture", "--model", help="Reconstruction model, e.g. mlx3d-capture."
    ),
    device: str = typer.Option("auto", "--device", help="auto | cpu | mps"),
    quality: str = typer.Option("fast", "--quality", help="mlx3d quality preset."),
    iters: int | None = typer.Option(None, "--iters", help="Override training iterations."),
    max_dim: int | None = typer.Option(None, "--max-dim", help="Max training image dimension."),
    sh_degree: int | None = typer.Option(None, "--sh-degree", help="Spherical harmonic degree."),
    poses: str = typer.Option("auto", "--poses", help="auto | colmap | builtin | existing"),
    refine_poses: str = typer.Option("auto", "--refine-poses", help="auto | on | off"),
    low_memory: bool = typer.Option(False, "--low-mem", help="Use mlx3d low-memory mode."),
    seed: int = typer.Option(0, "--seed", help="Random seed; <0 disables seeding."),
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
                f"[green]wrote[/green] {output} "
                f"({results[0].metadata.point_count:,} points)"
            )

    def _human(assets: list[Manifest]) -> None:
        for asset in assets:
            console.print(
                f"[green]gaussian[/green] {asset.id}  points={asset.metadata.point_count}"
            )

    report(results, _human)
