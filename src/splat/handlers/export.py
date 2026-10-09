import shutil
from dataclasses import dataclass, field
from pathlib import Path

from splat.domain.errors import SplatDomainError
from splat.domain.manifest import Manifest, ManifestKind
from splat.registry.wiring import get_manifest_repository, get_writer


@dataclass(frozen=True)
class ExportRequest:
    input: Manifest
    output_path: Path
    profile: str | None = None  # compression profile, gaussian clouds only
    pruning: str | None = None
    target_count: int | None = None


@dataclass
class ExportResult:
    path: Path
    point_count: int | None = None
    warnings: list[str] = field(default_factory=list)


def handle(request: ExportRequest) -> ExportResult:
    """Writes a manifest to a file the format of its extension, with a sidecar."""
    asset, output = request.input, request.output_path
    if request.profile is None and (request.pruning or request.target_count is not None):
        raise SplatDomainError("--pruning and --target-count need --profile.")
    if asset.kind == ManifestKind.GAUSSIAN_CLOUD:
        result = _export_cloud(request)
    else:
        if request.profile is not None:
            raise SplatDomainError("--profile only applies to gaussian clouds.")
        if output.suffix != asset.content_path.suffix:
            raise SplatDomainError(
                f"{asset.kind.value} {asset.id} exports as {asset.content_path.suffix}, "
                f"not {output.suffix}."
            )
        shutil.copyfile(asset.content_path, output)
        result = ExportResult(path=output)
    get_manifest_repository().write_sidecar(asset.id, output)
    return result


def _export_cloud(request: ExportRequest) -> ExportResult:
    cloud = request.input.as_gaussian_cloud()
    if request.profile is not None:
        from splat.adapters.compression.prune_quantize import PruneQuantizeCompressor

        cloud = PruneQuantizeCompressor().compress(
            cloud,
            profile=request.profile,
            pruning=request.pruning or "threshold",
            target_count=request.target_count,
        )
    writer = get_writer(request.output_path.suffix)
    warnings = writer.supports(cloud)
    writer.write(cloud, request.output_path)
    return ExportResult(path=request.output_path, point_count=cloud.point_count, warnings=warnings)
