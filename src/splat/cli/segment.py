from pathlib import Path

import typer

from splat.cli._console import console, error
from splat.cli._pipeline_io import report, resolve_inputs
from splat.domain.asset import AssetKind
from splat.domain.errors import SplatDomainError
from splat.handlers.segment import SegmentRequest
from splat.registry.wiring import get_asset_cache, get_client


def segment(
    input: str = typer.Argument(
        ..., help="Image path, @<asset-id>, or '-' to read piped asset records."
    ),
    output_dir: Path | None = typer.Option(
        None, "-o", "--output", help="Also export stickers here."
    ),
    model: str = typer.Option("sam-mlx", "--model"),
    max_stickers: int = typer.Option(20, "--max-stickers"),
    device: str = typer.Option("auto", "--device", help="auto | cpu | mps"),
) -> None:
    """Segment image(s) into RGBA sticker cutouts (cached; fans out to many assets)."""
    cache = get_asset_cache()
    try:
        inputs = resolve_inputs(input, cache, default_kind=AssetKind.IMAGE)
        all_stickers = get_client().segment(
            SegmentRequest(inputs=inputs, model=model, max_stickers=max_stickers, device=device)
        )
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, sticker_asset in enumerate(all_stickers):
            (output_dir / f"sticker_{i:03d}.png").write_bytes(
                sticker_asset.content_path.read_bytes()
            )

    def _human(assets: list) -> None:
        console.print(f"[green]segmented[/green] {len(assets)} stickers")
        for asset in assets:
            console.print(
                f"  {asset.id}  score={asset.metadata['score']:.3f}  bbox={asset.metadata['bbox']}"
            )

    report(all_stickers, _human)
