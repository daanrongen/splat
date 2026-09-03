from pathlib import Path

from splat.domain.image_space import Shape3D
from splat.ports.mesh import MeshPredictionBackend


class PredictMeshUseCase:
    def __init__(self, backend: MeshPredictionBackend) -> None:
        self._backend = backend

    def execute(self, image_path: Path, **params) -> Shape3D:
        return self._backend.predict(image_path, **params)
