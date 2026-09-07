from pathlib import Path

from splat.domain.gaussians import GaussianCloud
from splat.ports.compression import Compressor
from splat.ports.splat_io import SplatReader, SplatWriter


class CompressUseCase:
    """Deterministically prune and quantize a GaussianCloud for a delivery profile."""

    def __init__(self, reader: SplatReader, writer: SplatWriter, compressor: Compressor) -> None:
        self._reader = reader
        self._writer = writer
        self._compressor = compressor

    def execute(
        self,
        input_path: Path,
        output_path: Path,
        *,
        profile: str = "web-delivery",
        pruning: str = "threshold",
        target_count: int | None = None,
    ) -> GaussianCloud:
        cloud = self._reader.read(input_path)
        compressed = self._compressor.compress(
            cloud, profile=profile, pruning=pruning, target_count=target_count
        )
        self._writer.write(compressed, output_path)
        return compressed
