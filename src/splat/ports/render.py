from pathlib import Path
from typing import Protocol

from splat.domain.gaussians import GaussianCloud


class RenderBackend(Protocol):
    """Renders a GaussianCloud to a still image using some render engine."""

    name: str

    def render(self, cloud: GaussianCloud, output_path: Path, **params) -> None: ...
