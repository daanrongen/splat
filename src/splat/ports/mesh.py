"""Both directions `splat mesh` is concerned with: predicting a mesh from an
image(+depth) (`MeshPredictionBackend`), and exporting a Gaussian cloud to a
mesh file (`MeshExporter`, still deferred — SuGaR-style surface extraction
needs Open3D and a full surface-alignment pipeline, out of scope for the
MVP).
"""

from pathlib import Path
from typing import Protocol

from splat.domain.gaussians import GaussianCloud
from splat.domain.image_space import DepthMap, Shape3D
from splat.domain.value_objects import ModelLicense


class MeshExporter(Protocol):
    name: str

    def export(self, cloud: GaussianCloud, path: Path, *, format: str, **params) -> None: ...


class MeshPredictionBackend(Protocol):
    name: str
    license: ModelLicense

    def predict(
        self, image_path: Path, *, depth_map: DepthMap | None = None, **params
    ) -> Shape3D: ...
