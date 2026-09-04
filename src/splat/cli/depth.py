from pathlib import Path

import numpy as np
import typer
from PIL import Image

from splat.cli._console import console, error
from splat.cli._pipeline_io import report, resolve_inputs
from splat.domain.asset import AssetKind
from splat.domain.errors import SplatDomainError
from splat.handlers.depth import DepthRequest
from splat.registry.wiring import get_asset_cache, get_client


def depth(
    input: str = typer.Argument(
        ..., help="Image path, @<asset-id>, or '-' to read piped asset records."
    ),
    output: Path | None = typer.Option(
        None, "-o", "--output", help="Render a viewable (normalized) depth PNG here."
    ),
    model: str = typer.Option("depth-pro", "--model"),
    device: str = typer.Option("auto", "--device", help="auto | cpu | mps"),
) -> None:
    """Estimate per-pixel metric depth for image(s) (cached losslessly as .npy)."""
    cache = get_asset_cache()
    try:
        inputs = resolve_inputs(input, cache, default_kind=AssetKind.IMAGE)
        results = get_client().depth(DepthRequest(inputs=inputs, model=model, device=device))
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc

    if output is not None and results:
        depth_array = np.load(results[0].content_path)
        span = max(float(depth_array.max() - depth_array.min()), 1e-6)
        normalized = (depth_array - depth_array.min()) / span
        Image.fromarray((normalized * 255).astype(np.uint8)).save(output)

    def _human(assets: list) -> None:
        for asset in assets:
            console.print(
                f"[green]depth[/green] {asset.id}  "
                f"focal_length={asset.metadata.get('focal_length_px')}"
            )

    report(results, _human)
