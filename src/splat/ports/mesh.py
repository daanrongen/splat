"""Deterministic Gaussian-cloud-to-mesh export contract (`splat tools
extract-surface`, not yet implemented). Image-to-mesh prediction (TripoSR)
was removed as a top-level backend - see #54: it never had a
working implementation, and mesh prediction belongs behind `render`'s
sibling shape (a swappable model catalog) if/when a real backend lands,
not as a permanent stub cluttering the top-level surface.
"""

from pathlib import Path
from typing import Protocol

from splat.domain.gaussians import GaussianCloud


class MeshExporter(Protocol):
    name: str

    def export(self, cloud: GaussianCloud, path: Path, *, format: str, **params) -> None: ...
