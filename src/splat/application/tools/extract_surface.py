from dataclasses import dataclass
from pathlib import Path

from splat.ports.mesh import MeshExporter
from splat.ports.splat_io import SplatReader


@dataclass
class ExtractSurfaceResult:
    input_point_count: int
    vertex_count: int
    face_count: int


class ExtractSurfaceUseCase:
    """Extract a textured surface mesh from a GaussianCloud."""

    def __init__(self, reader: SplatReader, exporter: MeshExporter) -> None:
        self._reader = reader
        self._exporter = exporter

    def execute(
        self, input_path: Path, output_path: Path, *, format: str, **params
    ) -> ExtractSurfaceResult:
        cloud = self._reader.read(input_path)
        vertex_count, face_count = self._exporter.export(
            cloud, output_path, format=format, **params
        )
        return ExtractSurfaceResult(
            input_point_count=cloud.point_count,
            vertex_count=vertex_count,
            face_count=face_count,
        )
