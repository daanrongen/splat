"""Deferred port: no adapters ship yet. The optimization-based half of the
Gaussian Splatting ecosystem (3DGS, 2DGS, Scaffold-GS, ...) is CUDA-only with
no settled MPS story (gsplat-mlx, msplat, OpenSplat all take different,
immature approaches) — this seam exists so a future adapter has somewhere to
plug in, without committing to one now.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from splat.domain.gaussians import GaussianCloud
from splat.domain.value_objects import ModelLicense

if TYPE_CHECKING:
    from splat.ports.rasterizer import RasterizerBackend


class TrainingBackend(Protocol):
    name: str
    license: ModelLicense

    def train(
        self,
        dataset_dir: Path,
        *,
        rasterizer: "RasterizerBackend",
        iterations: int,
        device: str = "auto",
        **params,
    ) -> GaussianCloud: ...
