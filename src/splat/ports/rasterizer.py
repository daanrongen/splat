"""Deferred port: no adapters ship yet. See ports/training.py — a
RasterizerBackend is what a future TrainingBackend adapter would render
through during optimization.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from splat.domain.gaussians import GaussianCloud
from splat.domain.value_objects import Camera, Pose


@dataclass(frozen=True)
class RenderResult:
    rgb: np.ndarray  # (H, W, 3) float32
    alpha: np.ndarray  # (H, W) float32
    depth: np.ndarray | None = None  # (H, W) float32


class RasterizerBackend(Protocol):
    def render(self, cloud: GaussianCloud, camera: Camera, pose: Pose) -> RenderResult: ...
