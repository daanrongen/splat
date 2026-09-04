from pathlib import Path

from splat.application.tools.convert import ConvertResult
from splat.domain.asset import Asset
from splat.domain.gaussians import GaussianCloud
from splat.handlers import models as models_handler
from splat.handlers.depth import DepthRequest
from splat.handlers.depth import handle as handle_depth
from splat.handlers.diffuse import DiffuseRequest, DiffuseResult
from splat.handlers.diffuse import handle as handle_diffuse
from splat.handlers.gaussian import GaussianRequest, GaussianResult
from splat.handlers.gaussian import handle as handle_gaussian
from splat.handlers.inspect import info as handle_info
from splat.handlers.inspect import validate as handle_validate
from splat.handlers.mesh import MeshRequest
from splat.handlers.mesh import handle as handle_mesh
from splat.handlers.segment import SegmentRequest
from splat.handlers.segment import handle as handle_segment
from splat.handlers.tools.compress import CompressRequest
from splat.handlers.tools.compress import handle as handle_compress
from splat.handlers.tools.convert import ConvertRequest
from splat.handlers.tools.convert import handle as handle_convert
from splat.ports.client import InfoSummary, ModelInfo, ModelSummary, ValidationSummary


class LocalSplatClient:
    """In-process pass-through to handlers/*.py — the client used when
    SPLAT_URL isn't set."""

    def diffuse(self, request: DiffuseRequest) -> DiffuseResult:
        return handle_diffuse(request)

    def segment(self, request: SegmentRequest) -> list[Asset]:
        return handle_segment(request)

    def depth(self, request: DepthRequest) -> list[Asset]:
        return handle_depth(request)

    def mesh(self, request: MeshRequest) -> list[Asset]:
        return handle_mesh(request)

    def gaussian(self, request: GaussianRequest) -> GaussianResult:
        return handle_gaussian(request)

    def convert(self, request: ConvertRequest) -> ConvertResult:
        return handle_convert(request)

    def compress(self, request: CompressRequest) -> GaussianCloud:
        return handle_compress(request)

    def info(self, path: Path) -> InfoSummary:
        cloud = handle_info(path)
        return InfoSummary(
            format=cloud.metadata.source_format,
            points=cloud.point_count,
            sh_degree=cloud.sh_degree,
            bbox_min=cloud.means.min(axis=0).tolist(),
            bbox_max=cloud.means.max(axis=0).tolist(),
        )

    def validate(self, path: Path, *, strict: bool = False) -> ValidationSummary:
        result = handle_validate(path, strict=strict)
        return ValidationSummary(
            valid=not result.issues, issues=result.issues, points=result.cloud.point_count
        )

    def models_list(self) -> list[ModelSummary]:
        return [
            ModelSummary(
                name=descriptor.name,
                runtime=getattr(descriptor, "runtime", "-"),
                license=str(descriptor.license),
                cached=cached,
            )
            for descriptor, cached in models_handler.list_models()
        ]

    def models_pull(self, name: str) -> None:
        models_handler.pull(name)

    def models_info(self, name: str) -> ModelInfo:
        descriptor = models_handler.info(name)
        return ModelInfo(
            name=descriptor.name,
            runtime=getattr(descriptor, "runtime", "-"),
            source=descriptor.hf_repo_id,
            license=str(descriptor.license),
            min_images=getattr(descriptor, "min_images", None),
            max_images=getattr(descriptor, "max_images", None),
        )

    def models_rm(self, name: str) -> None:
        models_handler.rm(name)
