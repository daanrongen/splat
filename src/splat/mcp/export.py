from pathlib import Path

from splat.domain.manifest import ManifestKind
from splat.handlers.export import ExportRequest, handle
from splat.mcp._inputs import resolve_input_asset
from splat.registry.wiring import get_manifest_repository


def export(
    input: str,
    output_path: str,
    profile: str | None = None,
    pruning: str = "threshold",
    target_count: int | None = None,
) -> dict:
    """Write an asset (path or @<asset-id>) to output_path in the format of its extension,
    optionally compressing a Gaussian cloud with profile web-delivery or archival."""
    asset = resolve_input_asset(
        input, get_manifest_repository(), default_kind=ManifestKind.GAUSSIAN_CLOUD
    )
    result = handle(
        ExportRequest(
            input=asset,
            output_path=Path(output_path),
            profile=profile,
            pruning=pruning,
            target_count=target_count,
        )
    )
    return {
        "path": str(result.path),
        "point_count": result.point_count,
        "warnings": result.warnings,
    }
