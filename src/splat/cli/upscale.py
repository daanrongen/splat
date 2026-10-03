from pathlib import Path

import typer

from splat.cli._console import console, error
from splat.cli._pipeline_io import export_output, prepare_output, report, resolve_inputs
from splat.domain.errors import SplatDomainError
from splat.domain.manifest import ManifestKind
from splat.handlers.upscale import UpscaleRequest
from splat.registry.wiring import get_client, get_manifest_repository


def upscale(
    input: str = typer.Argument(
        ..., help="Image path, @<asset-id>, or '-' to read piped asset records."
    ),
    output: Path | None = typer.Option(None, "-o", "--output", help="Also export PNG here."),
    factor: int = typer.Option(
        4, "--factor", help="Native upscale factor: 2 or 4.", envvar="SPLAT_UPSCALE_FACTOR"
    ),
    model: str = typer.Option("realesrgan-mlx", "--model", envvar="SPLAT_UPSCALE_MODEL"),
    tile: int = typer.Option(
        0,
        "--tile",
        help="Tile size for large images; 0 disables tiling.",
        envvar="SPLAT_UPSCALE_TILE",
    ),
) -> None:
    """Upscale image or sticker assets with a model-backed super-resolution backend."""
    cache = get_manifest_repository()
    try:
        prepare_output(output)
        inputs = resolve_inputs(input, cache, default_kind=ManifestKind.IMAGE)
        results = get_client().upscale(
            UpscaleRequest(inputs=inputs, model=model, factor=factor, tile=tile)
        )
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc

    if output is not None and results:
        export_output(results[0], output, cache)

    def _human(assets: list) -> None:
        for asset in assets:
            console.print(
                f"[green]upscaled[/green] {asset.id}  "
                f"{asset.metadata.source_width}x{asset.metadata.source_height} -> "
                f"{asset.metadata.output_width}x{asset.metadata.output_height}"
            )

    report(results, _human)
