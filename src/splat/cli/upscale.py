from pathlib import Path

import typer

from splat.cli._console import console, error
from splat.cli._pipeline_io import report, resolve_inputs
from splat.domain.errors import SplatDomainError
from splat.domain.manifest import ManifestKind
from splat.handlers.upscale import UpscaleRequest
from splat.registry.wiring import get_asset_cache, get_client


def upscale(
    input: str = typer.Argument(
        ..., help="Image path, @<asset-id>, or '-' to read piped asset records."
    ),
    output: Path | None = typer.Option(None, "-o", "--output", help="Also export PNG here."),
    factor: int = typer.Option(4, "--factor", help="Native upscale factor: 2 or 4."),
    model: str = typer.Option("realesrgan-mlx", "--model"),
    tile: int = typer.Option(0, "--tile", help="Tile size for large images; 0 disables tiling."),
) -> None:
    """Upscale image or sticker assets with a model-backed super-resolution backend."""
    cache = get_asset_cache()
    try:
        inputs = resolve_inputs(input, cache, default_kind=ManifestKind.IMAGE)
        results = get_client().upscale(
            UpscaleRequest(inputs=inputs, model=model, factor=factor, tile=tile)
        )
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc

    if output is not None and results:
        output.write_bytes(results[0].content_path.read_bytes())

    def _human(assets: list) -> None:
        for asset in assets:
            console.print(
                f"[green]upscaled[/green] {asset.id}  "
                f"{asset.metadata.source_width}x{asset.metadata.source_height} -> "
                f"{asset.metadata.output_width}x{asset.metadata.output_height}"
            )

    report(results, _human)
