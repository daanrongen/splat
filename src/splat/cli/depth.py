from pathlib import Path

import numpy as np
import typer

from splat.adapters.formats.image import write_png
from splat.cli._console import console, error
from splat.cli._pipeline_io import report, resolve_inputs
from splat.domain.errors import SplatDomainError
from splat.domain.manifest import ManifestKind
from splat.handlers.depth import DepthRequest
from splat.registry.wiring import get_client, get_manifest_repository


def depth(
    input: str = typer.Argument(
        ..., help="Image path, @<asset-id>, or '-' to read piped asset records."
    ),
    output: Path | None = typer.Option(
        None, "-o", "--output", help="Render a viewable (normalized) depth PNG here."
    ),
    model: str = typer.Option("depth-pro", "--model", envvar="SPLAT_DEPTH_MODEL"),
    device: str = typer.Option(
        "auto", "--device", help="auto | cpu | mps", envvar="SPLAT_DEPTH_DEVICE"
    ),
) -> None:
    """Estimate per-pixel metric depth for image(s) (cached losslessly as .npy)."""
    cache = get_manifest_repository()
    try:
        inputs = resolve_inputs(input, cache, default_kind=ManifestKind.IMAGE)
        results = get_client().depth(DepthRequest(inputs=inputs, model=model, device=device))
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc

    if output is not None and results:
        depth_array = np.load(results[0].content_path)
        span = max(float(depth_array.max() - depth_array.min()), 1e-6)
        normalized = (depth_array - depth_array.min()) / span
        write_png(output, (normalized * 255).astype(np.uint8))

    def _human(assets: list) -> None:
        for asset in assets:
            console.print(
                f"[green]depth[/green] {asset.id}  focal_length={asset.metadata.focal_length_px}"
            )

    report(results, _human)
