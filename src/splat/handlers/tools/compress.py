from dataclasses import dataclass
from pathlib import Path

from splat.adapters.compression.prune_quantize import PruneQuantizeCompressor
from splat.application.tools.compress import CompressUseCase
from splat.domain.gaussians import GaussianCloud
from splat.registry.wiring import get_reader, get_writer


@dataclass(frozen=True)
class CompressRequest:
    input_path: Path
    output_path: Path
    profile: str = "web-delivery"
    pruning: str = "threshold"
    target_count: int | None = None


def handle(request: CompressRequest) -> GaussianCloud:
    reader = get_reader(request.input_path.suffix)
    writer = get_writer(request.output_path.suffix)
    return CompressUseCase(reader, writer, PruneQuantizeCompressor()).execute(
        request.input_path,
        request.output_path,
        profile=request.profile,
        pruning=request.pruning,
        target_count=request.target_count,
    )
