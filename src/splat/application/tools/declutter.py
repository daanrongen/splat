from pathlib import Path

from splat.domain.gaussians import GaussianCloud
from splat.ports.declutter import Declutterer
from splat.ports.splat_io import SplatReader, SplatWriter


class DeclutterUseCase:
    """Deterministically remove isolated floater Gaussians from a GaussianCloud."""

    def __init__(self, reader: SplatReader, writer: SplatWriter, declutterer: Declutterer) -> None:
        self._reader = reader
        self._writer = writer
        self._declutterer = declutterer

    def execute(
        self,
        input_path: Path,
        output_path: Path,
        *,
        k: int = 16,
        std_ratio: float = 2.0,
    ) -> GaussianCloud:
        cloud = self._reader.read(input_path)
        cleaned = self._declutterer.declutter(cloud, k=k, std_ratio=std_ratio)
        self._writer.write(cleaned, output_path)
        return cleaned
