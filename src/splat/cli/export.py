from pathlib import Path

import typer

from splat.adapters.compression.prune_quantize import PROFILES
from splat.cli._console import console, error, warn
from splat.cli._pipeline_io import prepare_output, resolve_inputs
from splat.domain.errors import SplatDomainError
from splat.domain.manifest import ManifestKind
from splat.handlers.export import ExportRequest, handle
from splat.registry.wiring import get_manifest_repository


def export(
    input: str = typer.Argument(
        ..., help="Any asset: path/@<asset-id>, or '-' for one piped record."
    ),
    output: Path = typer.Option(..., "-o", "--output", help="Destination; format from extension."),
    profile: str | None = typer.Option(
        None,
        "--profile",
        help=f"Gaussian clouds: compress first with {' | '.join(PROFILES)}.",
        envvar="SPLAT_EXPORT_PROFILE",
    ),
    pruning: str = typer.Option(
        "threshold",
        "--pruning",
        help="With --profile: threshold | blue-noise (needs --target-count).",
        envvar="SPLAT_EXPORT_PRUNING",
    ),
    target_count: int | None = typer.Option(
        None, "--target-count", help="Point count for --pruning blue-noise."
    ),
) -> None:
    """Write an asset to a file (.ply/.splat/.sog for clouds), with a lineage sidecar."""
    try:
        prepare_output(output)
        inputs = resolve_inputs(
            input, get_manifest_repository(), default_kind=ManifestKind.GAUSSIAN_CLOUD
        )
        if len(inputs) != 1:
            raise SplatDomainError(f"export writes one asset, got {len(inputs)}.")
        result = handle(
            ExportRequest(
                input=inputs[0],
                output_path=output,
                profile=profile,
                pruning=pruning,
                target_count=target_count,
            )
        )
    except (SplatDomainError, ValueError) as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc

    for warning in result.warnings:
        warn(warning)
    points = f" ({result.point_count:,} points)" if result.point_count is not None else ""
    console.print(f"[green]wrote[/green] {result.path}{points}")
