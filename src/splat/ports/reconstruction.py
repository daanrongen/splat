from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from splat.domain.gaussians import GaussianCloud
from splat.domain.value_objects import ModelLicense


class ReconstructionBackend(Protocol):
    """Feed-forward image(s) -> GaussianCloud, no per-scene optimization."""

    name: str
    license: ModelLicense

    provides_metric_scale: bool
    """True when the backend's output is already in real-world units.

    SfM and most feed-forward reconstruction have no absolute scale, so
    `run_gaussian` normalizes by default; a backend that predicts metres
    (SHARP) sets this so that normalization is skipped instead of throwing
    away the one thing it uniquely provides.
    """

    def reconstruct(
        self,
        images: list[Path],
        *,
        device: str = "auto",
        on_progress: Callable[[str], None] | None = None,
        **params,
    ) -> GaussianCloud: ...

    def required_image_count(self) -> tuple[int, int | None]:
        """(min, max) images accepted; max=None means unbounded.

        Implementations are classmethods — the count is a fact about the
        backend, not a loaded model — so `models info` and the gaussian
        handler's contract can both read it without pulling weights.
        """
        ...
