from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import typer

from splat.cli._console import console, error, warn
from splat.cli._pipeline_io import export_output, is_piped, prepare_output, report, resolve_inputs
from splat.domain.errors import SplatDomainError
from splat.domain.manifest import ManifestKind
from splat.handlers.render import RenderRequest, handle
from splat.registry.wiring import get_manifest_repository


@contextmanager
def _progress() -> Iterator[Callable[[str], None] | None]:
    """Blender reports device, then scene sync, then `Sample n/m`. Cycles can
    also spend minutes compiling Metal kernels on first use, which is the part
    that reads as a hang, so it goes on one live line rather than nowhere.
    Suppressed when piped, where stdout is a machine-readable record."""
    if is_piped():
        yield None
        return

    with console.status("[dim]starting render[/dim]") as status:

        def on_progress(message: str) -> None:
            if message.startswith("device "):
                device = message.removeprefix("device ")
                if device.startswith("CPU"):
                    warn(f"cycles is rendering on {device}; expect this to be slow")
                else:
                    console.print(f"[dim]render device:[/dim] {device}")
                return
            status.update(f"[dim]{message}[/dim]")

        yield on_progress


def render(
    input: str = typer.Argument(
        ..., help="Gaussian cloud path/@<asset-id>, or '-' to read piped asset records."
    ),
    output: Path | None = typer.Option(None, "-o", "--output", help="Write the rendered PNG here."),
    model: str = typer.Option(
        "blender", "--model", help="Render backend.", envvar="SPLAT_RENDER_MODEL"
    ),
    width: int = typer.Option(
        1280, "--width", help="Render width in pixels.", envvar="SPLAT_RENDER_WIDTH"
    ),
    height: int = typer.Option(
        720, "--height", help="Render height in pixels.", envvar="SPLAT_RENDER_HEIGHT"
    ),
    samples: int = typer.Option(
        32, "--samples", help="Render samples (backend-dependent).", envvar="SPLAT_RENDER_SAMPLES"
    ),
    engine: str = typer.Option(
        "cycles",
        "--engine",
        help="cycles (accurate alpha-composited kernels) | eevee (fast preview).",
        envvar="SPLAT_RENDER_ENGINE",
    ),
    background: str = typer.Option(
        "black",
        "--background",
        help="Backdrop: transparent, black, white, grey, or a hex colour.",
        envvar="SPLAT_RENDER_BACKGROUND",
    ),
    azimuth: float | None = typer.Option(
        None,
        "--azimuth",
        help="Orbit angle in degrees around the up axis (default 25).",
        envvar="SPLAT_RENDER_AZIMUTH",
    ),
    elevation: float | None = typer.Option(
        None,
        "--elevation",
        help="Orbit angle in degrees above the horizon (default 20).",
        envvar="SPLAT_RENDER_ELEVATION",
    ),
    distance: float | None = typer.Option(
        None,
        "--distance",
        help="Camera distance from the look-at point (default fits the cloud).",
        envvar="SPLAT_RENDER_DISTANCE",
    ),
    fov: float | None = typer.Option(
        None,
        "--fov",
        help="Horizontal field of view in degrees.",
        envvar="SPLAT_RENDER_FOV",
    ),
    look_at: str | None = typer.Option(
        None,
        "--look-at",
        help="Orbit centre as 'x,y,z' (default the cloud's median point).",
        envvar="SPLAT_RENDER_LOOK_AT",
    ),
) -> None:
    """Render a Gaussian splat to a still image."""
    cache = get_manifest_repository()
    request = RenderRequest(
        inputs=[],
        model=model,
        width=width,
        height=height,
        samples=samples,
        engine=engine,
        background=background,
        azimuth=azimuth,
        elevation=elevation,
        distance=distance,
        fov=fov,
        look_at=look_at,
    )
    try:
        prepare_output(output)
        inputs = resolve_inputs(input, cache, default_kind=ManifestKind.GAUSSIAN_CLOUD)
        with _progress() as on_progress:
            results = handle(replace(request, inputs=inputs), on_progress=on_progress)
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc

    if output is not None and results:
        export_output(results[0], output, cache)

    def _human(assets: list) -> None:
        for asset in assets:
            console.print(
                f"[green]render[/green] {asset.id}  "
                f"{asset.metadata.output_width}x{asset.metadata.output_height}"
            )

    report(results, _human)
