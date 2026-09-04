"""Mesh-shaped contracts: model-backed image-to-mesh prediction and deterministic
Gaussian-cloud export.
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
