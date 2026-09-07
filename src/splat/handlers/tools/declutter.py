from dataclasses import dataclass
from pathlib import Path

from splat.adapters.cleanup.density_declutter import DensityDeclutterer
from splat.application.tools.declutter import DeclutterUseCase
from splat.domain.gaussians import GaussianCloud
from splat.registry.wiring import get_reader, get_writer


@dataclass(frozen=True)
class DeclutterRequest:
    input_path: Path
    output_path: Path
    k: int = 16
    std_ratio: float = 2.0


def handle(request: DeclutterRequest) -> GaussianCloud:
    reader = get_reader(request.input_path.suffix)
    writer = get_writer(request.output_path.suffix)
    return DeclutterUseCase(reader, writer, DensityDeclutterer()).execute(
        request.input_path,
        request.output_path,
        k=request.k,
        std_ratio=request.std_ratio,
    )
