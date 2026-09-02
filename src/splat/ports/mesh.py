"""Deferred port: no adapters ship yet (SuGaR-style mesh extraction needs
Open3D and a full surface-alignment pipeline — out of scope for the MVP)."""

from pathlib import Path
from typing import Protocol

from splat.domain.gaussians import GaussianCloud


class MeshExporter(Protocol):
    name: str

    def export(self, cloud: GaussianCloud, path: Path, *, format: str, **params) -> None: ...
