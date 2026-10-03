from pathlib import Path

import typer

from splat.cli._console import console, error, warn
from splat.cli._pipeline_io import is_piped, prepare_output, report, resolve_inputs
from splat.domain.errors import SplatDomainError
from splat.domain.manifest import Manifest, ManifestKind
from splat.handlers.gaussian import GaussianRequest
from splat.handlers.gaussian import handle as handle_gaussian
from splat.registry.wiring import get_client, get_manifest_repository, get_reader, get_writer


def _export_gaussian(asset: Manifest, output: Path) -> list[str]:
    cloud = get_reader(asset.content_path.suffix).read(asset.content_path)
    writer = get_writer(output.suffix)
    warnings = writer.supports(cloud)
    writer.write(cloud, output)
    get_manifest_repository().write_sidecar(asset.id, output)
    return warnings


_REGISTRATION_WARN_FRACTION = 0.5


def _check_registration(asset: Manifest, input_count: int, min_registered: float | None) -> None:
    """SfM can silently register only a fraction of its input views and keep
    going - training on that fraction still "succeeds", just on much less
    data than the point count suggests. Warn (or, under --min-registered,
    fail) so that isn't discovered later by reading the cloud."""
    registered = asset.metadata.capture_camera_count
    if registered is None or input_count == 0:
        return
    fraction = registered / input_count
    message = (
        f"reconstruction registered {registered} of {input_count} input images "
        f"({fraction:.0%}); quality is likely lower than the point count suggests"
    )
    if min_registered is not None and fraction < min_registered:
        error(message)
        raise typer.Exit(code=1)
    if fraction < _REGISTRATION_WARN_FRACTION:
        warn(message)


def gaussian(
    inputs: list[str] = typer.Argument(
        ..., help="Image paths, @<asset-id>s, or '-' to read piped asset records."
    ),
    output: Path | None = typer.Option(None, "-o", "--output", help="Output splat file path."),
    model: str = typer.Option(
        "sharp",
        "--model",
        help="Reconstruction model, e.g. sharp or mlx3d-capture.",
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
    focal_35mm: float = typer.Option(
        30.0,
        "--focal-35mm",
        help="sharp: 35mm-equivalent focal length assumed for images without EXIF.",
        envvar="SPLAT_GAUSSIAN_FOCAL_35MM",
    ),
    min_registered: float | None = typer.Option(
        None,
        "--min-registered",
        help=(
            "Fail instead of warning if the fraction of input images the "
            "backend registered falls below this (0-1)."
        ),
        envvar="SPLAT_GAUSSIAN_MIN_REGISTERED",
    ),
    orbit_frames: int | None = typer.Option(
        None,
        "--orbit-frames",
        help=(
            "Render this many synthetic views of the reconstructed cloud, swept "
            "across --orbit-degrees, e.g. as multi-view input for a second "
            "`--model mlx3d-capture` pass."
        ),
        envvar="SPLAT_GAUSSIAN_ORBIT_FRAMES",
    ),
    orbit_degrees: float = typer.Option(
        30.0,
        "--orbit-degrees",
        help="Total azimuth sweep in degrees for --orbit-frames.",
        envvar="SPLAT_GAUSSIAN_ORBIT_DEGREES",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help=(
            "Print the reconstruction backend's own progress. Runs locally "
            "rather than through SPLAT_URL, same as `render`."
        ),
    ),
) -> None:
    """Reconstruct a Gaussian splat from images."""
    cache = get_manifest_repository()
    try:
        prepare_output(output)
        assets: list[Manifest] = []
        for input_arg in inputs:
            assets.extend(resolve_inputs(input_arg, cache, default_kind=ManifestKind.IMAGE))
        request = GaussianRequest(
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
            focal_35mm=focal_35mm,
            orbit_frames=orbit_frames,
            orbit_degrees=orbit_degrees,
        )
        if verbose:
            results = handle_gaussian(
                request, on_progress=lambda message: console.print(f"[dim]{message}[/dim]")
            )
        else:
            results = get_client().gaussian(request)
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc

    _check_registration(results[0], len(assets), min_registered)

    if output is not None and results:
        for warning in _export_gaussian(results[0], output):
            warn(warning)
        if not is_piped():
            console.print(
                f"[green]wrote[/green] {output} ({results[0].metadata.point_count:,} points)"
            )

    def _human(assets: list[Manifest]) -> None:
        for asset in assets:
            if asset.kind is ManifestKind.GAUSSIAN_CLOUD:
                console.print(
                    f"[green]gaussian[/green] {asset.id}  points={asset.metadata.point_count}"
                )
            else:
                console.print(f"[green]orbit frame[/green] {asset.id}  parent={results[0].id}")

    report(results, _human)
