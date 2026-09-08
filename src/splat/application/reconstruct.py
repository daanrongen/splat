from collections.abc import Callable
from pathlib import Path

from splat.domain.gaussians import GaussianCloud
from splat.ports.reconstruction import ReconstructionBackend


class ReconstructUseCase:
    """Image(s) -> GaussianCloud via a feed-forward ReconstructionBackend."""

    def __init__(self, backend: ReconstructionBackend) -> None:
        self._backend = backend

    def execute(
        self,
        images: list[Path],
        *,
        device: str = "auto",
        on_progress: Callable[[str], None] | None = None,
        **params,
    ) -> GaussianCloud:
        return self._backend.reconstruct(images, device=device, on_progress=on_progress, **params)
