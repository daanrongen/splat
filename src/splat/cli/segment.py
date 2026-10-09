from pathlib import Path

import typer

from splat.cli._console import console, error
from splat.cli._pipeline_io import export_output, report, resolve_inputs
from splat.domain.errors import SplatDomainError
from splat.domain.manifest import ManifestKind
from splat.handlers.segment import SegmentRequest
from splat.registry.wiring import get_client, get_manifest_repository


def segment(
    input: str = typer.Argument(
        ..., help="Image path, @<asset-id>, or '-' to read piped asset records."
    ),
    output: Path | None = typer.Option(
        None, "-o", "--output", help="Export stickers to this directory, or one sticker to a .png."
    ),
    model: str = typer.Option("sam-mlx", "--model", envvar="SPLAT_SEGMENT_MODEL"),
    max_stickers: int = typer.Option(20, "--max-stickers", envvar="SPLAT_SEGMENT_MAX_STICKERS"),
    device: str = typer.Option(
        "auto", "--device", help="auto | cpu | mps", envvar="SPLAT_SEGMENT_DEVICE"
    ),
    point: list[str] = typer.Option(
        [], "--point", help="Foreground point 'x,y' in pixels; repeatable."
    ),
    not_point: list[str] = typer.Option(
        [], "--not-point", help="Background point 'x,y' in pixels; repeatable."
    ),
    box: str | None = typer.Option(None, "--box", help="Box 'x0,y0,x1,y1' in pixels."),
    foreground: bool = typer.Option(
        False, "--foreground", help="Merge the masks into one cutout of the main subject."
    ),
    drop_background: bool = typer.Option(
        False, "--drop-background", help="Drop masks that cover the frame or its border."
    ),
) -> None:
    """Segment image(s) into RGBA sticker cutouts (cached; fans out to many assets)."""
    cache = get_manifest_repository()
    try:
        inputs = resolve_inputs(input, cache, default_kind=ManifestKind.IMAGE)
        all_stickers = get_client().segment(
            SegmentRequest(
                inputs=inputs,
                model=model,
                max_stickers=max_stickers,
                device=device,
                foreground=foreground,
                drop_background=drop_background,
                points=(*point, *(f"{p},0" for p in not_point)),
                box=box,
            )
        )
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc

    if output is not None and output.suffix:
        if len(all_stickers) != 1:
            error(f"-o {output} needs exactly one sticker, got {len(all_stickers)}")
            raise typer.Exit(code=1)
        export_output(all_stickers[0], output, cache)
    elif output is not None:
        output.mkdir(parents=True, exist_ok=True)
        for i, sticker_asset in enumerate(all_stickers):
            export_output(sticker_asset, output / f"sticker_{i:03d}.png", cache)

    def _human(assets: list) -> None:
        console.print(f"[green]segmented[/green] {len(assets)} stickers")
        for asset in assets:
            console.print(
                f"  {asset.id}  score={asset.metadata.score:.3f}  bbox={asset.metadata.bbox}"
            )

    report(all_stickers, _human)
