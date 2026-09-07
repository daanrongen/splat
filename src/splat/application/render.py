from pathlib import Path

from splat.domain.gaussians import GaussianCloud
from splat.ports.render import RenderBackend


class RenderUseCase:
    def __init__(self, backend: RenderBackend) -> None:
        self._backend = backend

    def execute(self, cloud: GaussianCloud, output_path: Path, **params) -> None:
        self._backend.render(cloud, output_path, **params)
