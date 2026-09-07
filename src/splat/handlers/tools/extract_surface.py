from dataclasses import dataclass
from pathlib import Path

from splat.adapters.mesh.poisson import PoissonMeshExporter
from splat.application.tools.extract_surface import ExtractSurfaceResult, ExtractSurfaceUseCase
from splat.registry.wiring import get_reader


@dataclass(frozen=True)
class ExtractSurfaceRequest:
    input_path: Path
    output_path: Path
    format: str | None = None
    depth: int = 9
    opacity_threshold: float = 0.1


def handle(request: ExtractSurfaceRequest) -> ExtractSurfaceResult:
    reader = get_reader(request.input_path.suffix)
    fmt = request.format or request.output_path.suffix.lstrip(".")
    return ExtractSurfaceUseCase(reader, PoissonMeshExporter()).execute(
        request.input_path,
        request.output_path,
        format=fmt,
        depth=request.depth,
        opacity_threshold=request.opacity_threshold,
    )
