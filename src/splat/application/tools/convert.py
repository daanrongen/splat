from dataclasses import dataclass
from pathlib import Path

from splat.domain.gaussians import GaussianCloud
from splat.ports.splat_io import SplatReader, SplatWriter


@dataclass
class ConvertResult:
    cloud: GaussianCloud
    warnings: list[str]


class ConvertUseCase:
    """Deterministically rewrite a GaussianCloud from one file format to another."""

    def __init__(self, reader: SplatReader, writer: SplatWriter) -> None:
        self._reader = reader
        self._writer = writer

    def execute(self, input_path: Path, output_path: Path) -> ConvertResult:
        cloud = self._reader.read(input_path)
        warnings = self._writer.supports(cloud)
        self._writer.write(cloud, output_path)
        return ConvertResult(cloud=cloud, warnings=warnings)
