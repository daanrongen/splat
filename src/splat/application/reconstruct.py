from pathlib import Path

from splat.domain.gaussians import GaussianCloud
from splat.ports.reconstruction import ReconstructionBackend


class ReconstructUseCase:
    """Image(s) -> GaussianCloud via a feed-forward ReconstructionBackend."""

    def __init__(self, backend: ReconstructionBackend) -> None:
        self._backend = backend

    def execute(self, images: list[Path], *, device: str = "auto", **params) -> GaussianCloud:
        return self._backend.reconstruct(images, device=device, **params)
