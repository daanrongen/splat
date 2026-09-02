from pathlib import Path
from typing import Protocol

from splat.domain.gaussians import GaussianCloud


class SplatReader(Protocol):
    """Parses a file into the canonical GaussianCloud aggregate."""

    def read(self, path: Path) -> GaussianCloud: ...


class SplatWriter(Protocol):
    """Serializes a GaussianCloud to a file in one concrete format."""

    def write(self, cloud: GaussianCloud, path: Path) -> None: ...

    def supports(self, cloud: GaussianCloud) -> list[str]:
        """Warnings for lossy downgrades this writer will perform on `cloud`
        (e.g. "SH degree 3 -> 0"). Empty list means a lossless write."""
        ...
