from dataclasses import dataclass
from pathlib import Path

from splat.application.reconstruct import ReconstructUseCase
from splat.domain.errors import SplatDomainError
from splat.domain.gaussians import GaussianCloud
from splat.registry.wiring import get_model_source, get_reconstruction_backend, get_writer


@dataclass(frozen=True)
class GaussianRequest:
    inputs: list[Path]
    output_path: Path
    model: str = "mvsplat"
    device: str = "auto"


@dataclass(frozen=True)
class GaussianResult:
    cloud: GaussianCloud
    warnings: list[str]


def handle(request: GaussianRequest) -> GaussianResult:
    model_source = get_model_source()
    backend = get_reconstruction_backend(
        request.model, model_source=model_source, device=request.device
    )

    min_images, max_images = backend.required_image_count()
    if len(request.inputs) < min_images or (
        max_images is not None and len(request.inputs) > max_images
    ):
        upper = max_images if max_images is not None else "∞"
        raise SplatDomainError(
            f"Model {request.model!r} requires between {min_images} and {upper} images, "
            f"got {len(request.inputs)}."
        )

    cloud = ReconstructUseCase(backend).execute(request.inputs, device=request.device)
    writer = get_writer(request.output_path.suffix)
    warnings = writer.supports(cloud)
    writer.write(cloud, request.output_path)
    return GaussianResult(cloud=cloud, warnings=warnings)
