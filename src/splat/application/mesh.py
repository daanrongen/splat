from pathlib import Path

from splat.domain.image_space import DepthMap, Shape3D
from splat.ports.mesh import MeshPredictionBackend


class PredictMeshUseCase:
    def __init__(self, backend: MeshPredictionBackend) -> None:
        self._backend = backend

    def execute(self, image_path: Path, *, depth_map: DepthMap | None = None, **params) -> Shape3D:
        return self._backend.predict(image_path, depth_map=depth_map, **params)
