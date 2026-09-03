"""Both directions `mesh`-shaped work takes in this project: predicting a
mesh from an image via a learned model (`MeshPredictionBackend`, the `splat
mesh` command), and exporting a Gaussian cloud to a mesh file (`MeshExporter`,
`splat tools extract.surface` — still undiscovered by any registry, same as
`ports/training.py`/`ports/rasterizer.py`; SuGaR-style surface extraction
needs Open3D and a full surface-alignment pipeline, out of scope for the
MVP).
"""

from pathlib import Path
from typing import Protocol

from splat.domain.gaussians import GaussianCloud
from splat.domain.image_space import Shape3D
from splat.domain.value_objects import ModelLicense


class MeshExporter(Protocol):
    name: str

    def export(self, cloud: GaussianCloud, path: Path, *, format: str, **params) -> None: ...


class MeshPredictionBackend(Protocol):
    name: str
    license: ModelLicense

    def predict(self, image_path: Path, **params) -> Shape3D: ...
