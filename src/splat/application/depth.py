from pathlib import Path

from splat.domain.image_space import DepthMap
from splat.ports.depth import DepthEstimationBackend


class EstimateDepthUseCase:
    def __init__(self, backend: DepthEstimationBackend) -> None:
        self._backend = backend

    def execute(self, image_path: Path, **params) -> DepthMap:
        return self._backend.estimate(image_path, **params)
