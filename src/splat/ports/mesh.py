"""Deterministic Gaussian-cloud-to-mesh export contract (`splat tools
extract.surface`). Image-to-mesh prediction (TripoSR) was removed as a
top-level backend - see #54: it never had a working implementation, and
mesh prediction belongs behind `render`'s sibling shape (a swappable model
catalog) if/when a real backend lands, not as a permanent stub cluttering
the top-level surface. `extract.surface` is a single fixed algorithm
(Poisson reconstruction) rather than a swappable catalog, so it lives here
as one concrete port, not a registry.
"""

from pathlib import Path
from typing import Protocol

from splat.domain.gaussians import GaussianCloud


class MeshExporter(Protocol):
    name: str

    def export(self, cloud: GaussianCloud, path: Path, *, format: str, **params) -> tuple[int, int]:
        """Write a mesh to `path` and return (vertex_count, face_count)."""
        ...
