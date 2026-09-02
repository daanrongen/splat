from pathlib import Path
from typing import Protocol

from splat.domain.gaussians import GaussianCloud
from splat.domain.value_objects import ModelLicense


class ReconstructionBackend(Protocol):
    """Feed-forward image(s) -> GaussianCloud, no per-scene optimization."""

    name: str
    license: ModelLicense

    def reconstruct(
        self, images: list[Path], *, device: str = "auto", **params
    ) -> GaussianCloud: ...

    def required_image_count(self) -> tuple[int, int | None]:
        """(min, max) images accepted; max=None means unbounded."""
        ...
