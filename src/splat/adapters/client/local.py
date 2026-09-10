from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from splat.application.models_admin import image_count_range, model_source_label, model_stage
from splat.ports.client import InfoSummary, ModelInfo, ModelSummary, ValidationSummary

if TYPE_CHECKING:
    from splat.application.tools.convert import ConvertResult
    from splat.application.tools.extract_surface import ExtractSurfaceResult
    from splat.domain.gaussians import GaussianCloud
    from splat.domain.manifest import Manifest
    from splat.handlers.caption import CaptionRequest
    from splat.handlers.depth import DepthRequest
    from splat.handlers.diffuse import DiffuseRequest, DiffuseResult
    from splat.handlers.embed import EmbedRequest
    from splat.handlers.gaussian import GaussianRequest
    from splat.handlers.segment import SegmentRequest
    from splat.handlers.tools.compress import CompressRequest
    from splat.handlers.tools.convert import ConvertRequest
    from splat.handlers.tools.declutter import DeclutterRequest
    from splat.handlers.tools.extract_surface import ExtractSurfaceRequest
    from splat.handlers.upscale import UpscaleRequest


class LocalSplatClient:
    """In-process pass-through to handlers/*.py — the client used when
    SPLAT_URL isn't set."""

    def diffuse(self, request: DiffuseRequest) -> DiffuseResult:
        from splat.handlers.diffuse import handle as handle_diffuse

        return handle_diffuse(request)

    def segment(self, request: SegmentRequest) -> list[Manifest]:
        from splat.handlers.segment import handle as handle_segment

        return handle_segment(request)

    def caption(self, request: CaptionRequest) -> list[Manifest]:
        from splat.handlers.caption import handle as handle_caption

        return handle_caption(request)

    def depth(self, request: DepthRequest) -> list[Manifest]:
        from splat.handlers.depth import handle as handle_depth

        return handle_depth(request)

    def upscale(self, request: UpscaleRequest) -> list[Manifest]:
        from splat.handlers.upscale import handle as handle_upscale

        return handle_upscale(request)

    def embed(self, request: EmbedRequest) -> list[Manifest]:
        from splat.handlers.embed import handle as handle_embed

        return handle_embed(request)

    def gaussian(self, request: GaussianRequest) -> list[Manifest]:
        from splat.handlers.gaussian import handle as handle_gaussian

        return handle_gaussian(request)

    def tools_convert(self, request: ConvertRequest) -> ConvertResult:
        from splat.handlers.tools.convert import handle as handle_convert

        return handle_convert(request)

    def tools_compress(self, request: CompressRequest) -> GaussianCloud:
        from splat.handlers.tools.compress import handle as handle_compress

        return handle_compress(request)

    def tools_declutter(self, request: DeclutterRequest) -> GaussianCloud:
        from splat.handlers.tools.declutter import handle as handle_declutter

        return handle_declutter(request)

    def tools_extract_surface(self, request: ExtractSurfaceRequest) -> ExtractSurfaceResult:
        from splat.handlers.tools.extract_surface import handle as handle_extract_surface

        return handle_extract_surface(request)

    def info(self, path: Path) -> InfoSummary:
        from splat.handlers.inspect import info as handle_info

        cloud = handle_info(path)
        return InfoSummary(
            format=cloud.metadata.source_format,
            points=cloud.point_count,
            sh_degree=cloud.sh_degree,
            bbox_min=cloud.means.min(axis=0).tolist(),
            bbox_max=cloud.means.max(axis=0).tolist(),
            coordinate_convention=cloud.metadata.coordinate_convention,
            up_axis=cloud.metadata.up_axis,
            source_model=cloud.metadata.source_model,
            license=cloud.metadata.license.spdx_id if cloud.metadata.license else None,
            capture_camera_count=cloud.metadata.capture_camera_count,
        )

    def validate(self, path: Path, *, strict: bool = False) -> ValidationSummary:
        from splat.handlers.inspect import validate as handle_validate

        result = handle_validate(path, strict=strict)
        return ValidationSummary(
            valid=not result.issues, issues=result.issues, points=result.cloud.point_count
        )

    def models_list(self) -> list[ModelSummary]:
        from splat.handlers import models as models_handler

        return [
            ModelSummary(
                name=descriptor.name,
                stage=model_stage(descriptor.name),
                runtime=getattr(descriptor, "runtime", "-"),
                license=descriptor.license.spdx_id,
                commercial=descriptor.license.is_commercial,
                cached=cached,
            )
            for descriptor, cached in models_handler.list_models()
        ]

    def models_pull(self, name: str) -> None:
        from splat.handlers import models as models_handler

        models_handler.pull(name)

    def models_info(self, name: str) -> ModelInfo:
        from splat.handlers import models as models_handler

        descriptor = models_handler.info(name)
        min_images, max_images = image_count_range(descriptor)
        return ModelInfo(
            name=descriptor.name,
            runtime=getattr(descriptor, "runtime", "-"),
            source=model_source_label(descriptor),
            license=str(descriptor.license),
            min_images=min_images,
            max_images=max_images,
            dimension=getattr(descriptor, "dimension", None),
            normalized=getattr(descriptor, "normalized", None),
            notes=getattr(descriptor, "notes", ""),
        )

    def models_rm(self, name: str) -> None:
        from splat.handlers import models as models_handler

        models_handler.rm(name)
