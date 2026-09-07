from pathlib import Path

import typer

from splat.cli._console import console, error
from splat.cli._pipeline_io import report, resolve_inputs
from splat.domain.errors import SplatDomainError
from splat.domain.manifest import ManifestKind
from splat.handlers.render import RenderRequest, handle
from splat.registry.wiring import get_manifest_repository


def render(
    input: str = typer.Argument(
        ..., help="Gaussian cloud path/@<asset-id>, or '-' to read piped asset records."
    ),
    output: Path | None = typer.Option(None, "-o", "--output", help="Write the rendered PNG here."),
    model: str = typer.Option("blender", "--model", help="Render backend."),
    width: int = typer.Option(1280, "--width", help="Render width in pixels."),
    height: int = typer.Option(720, "--height", help="Render height in pixels."),
    samples: int = typer.Option(32, "--samples", help="Render samples (backend-dependent)."),
    engine: str = typer.Option(
        "cycles",
        "--engine",
        help="cycles (accurate alpha-blended ellipsoids) | eevee (fast preview).",
    ),
) -> None:
    """Render a Gaussian splat to a still image."""
    cache = get_manifest_repository()
    try:
        inputs = resolve_inputs(input, cache, default_kind=ManifestKind.GAUSSIAN_CLOUD)
        results = handle(
            RenderRequest(
                inputs=inputs,
                model=model,
                width=width,
                height=height,
                samples=samples,
                engine=engine,
            )
        )
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc

    if output is not None and results:
        output.write_bytes(results[0].content_path.read_bytes())

    def _human(assets: list) -> None:
        for asset in assets:
            console.print(
                f"[green]render[/green] {asset.id}  "
                f"{asset.metadata.output_width}x{asset.metadata.output_height}"
            )

    report(results, _human)
