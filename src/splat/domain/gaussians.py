"""The canonical Gaussian aggregate — the hub every format adapter and every
reconstruction/compression adapter reads and writes, instead of N^2 pairwise
converters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from splat.domain.errors import InvalidGaussianCloud, UnsupportedSHDegree
from splat.domain.value_objects import ModelLicense

ScaleActivation = Literal["log", "linear"]
OpacityActivation = Literal["logit", "linear"]

MAX_SH_DEGREE = 3


def sh_rest_count(sh_degree: int) -> int:
    """Number of higher-order SH coefficients per channel for a given degree."""
    if sh_degree not in range(MAX_SH_DEGREE + 1):
        raise UnsupportedSHDegree(f"sh_degree must be 0..{MAX_SH_DEGREE}, got {sh_degree}")
    return (sh_degree + 1) ** 2 - 1


@dataclass
class GaussianCloudMetadata:
    source_format: str | None = None
    source_model: str | None = None
    license: ModelLicense | None = None
    up_axis: Literal["y", "z"] = "y"
    coordinate_convention: str = "opengl"

    @property
    def point_count(self) -> int:
        # Filled in by GaussianCloud.__post_init__; default until then.
        return getattr(self, "_point_count", 0)


@dataclass
class GaussianCloud:
    """Aggregate root for a set of 3D Gaussians.

    All arrays share a leading dimension N (the point count). Scale and
    opacity are stored in whatever parameterization the source used
    (`scale_activation`/`opacity_activation`) rather than being silently
    normalized on load — callers use `to_linear_scales()` /
    `to_activated_opacities()` to get "real" values, which keeps every
    adapter's sigmoid/exp handling in one place instead of re-derived per
    adapter.
    """

    means: np.ndarray  # (N, 3) float32
    scales: np.ndarray  # (N, 3) float32
    rotations: np.ndarray  # (N, 4) float32, quaternion (w, x, y, z)
    opacities: np.ndarray  # (N,) float32
    sh_dc: np.ndarray  # (N, 3) float32 — degree-0 SH (base color)
    sh_rest: np.ndarray | None = None  # (N, K, 3) float32, K = sh_rest_count(sh_degree)
    sh_degree: int = 0

    scale_activation: ScaleActivation = "log"
    opacity_activation: OpacityActivation = "logit"

    extra_features: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: GaussianCloudMetadata = field(default_factory=GaussianCloudMetadata)

    def __post_init__(self) -> None:
        n = self.means.shape[0]

        def check_shape(name: str, arr: np.ndarray, trailing: tuple[int, ...]) -> None:
            expected = (n, *trailing)
            if arr.shape != expected:
                raise InvalidGaussianCloud(f"{name} must have shape {expected}, got {arr.shape}")

        check_shape("means", self.means, (3,))
        check_shape("scales", self.scales, (3,))
        check_shape("rotations", self.rotations, (4,))
        check_shape("opacities", self.opacities, ())
        check_shape("sh_dc", self.sh_dc, (3,))

        if self.sh_degree not in range(MAX_SH_DEGREE + 1):
            raise UnsupportedSHDegree(f"sh_degree must be 0..{MAX_SH_DEGREE}, got {self.sh_degree}")

        expected_k = sh_rest_count(self.sh_degree)
        if expected_k == 0:
            if self.sh_rest is not None:
                raise InvalidGaussianCloud("sh_rest must be None when sh_degree == 0")
        else:
            if self.sh_rest is None:
                raise InvalidGaussianCloud(f"sh_rest is required for sh_degree={self.sh_degree}")
            check_shape("sh_rest", self.sh_rest, (expected_k, 3))

        if not np.all(np.isfinite(self.scales)):
            raise InvalidGaussianCloud("scales contains NaN/Inf")
        if not np.all(np.isfinite(self.opacities)):
            raise InvalidGaussianCloud("opacities contains NaN/Inf")

        self.metadata._point_count = n

    @property
    def point_count(self) -> int:
        return self.means.shape[0]

    def to_linear_scales(self) -> np.ndarray:
        """Scale in real (non-log) units, regardless of source parameterization."""
        if self.scale_activation == "log":
            return np.exp(self.scales)
        return self.scales

    def to_activated_opacities(self) -> np.ndarray:
        """Opacity in [0, 1], regardless of source parameterization."""
        if self.opacity_activation == "logit":
            return 1.0 / (1.0 + np.exp(-self.opacities))
        return self.opacities

    def to_log_scales(self) -> np.ndarray:
        """Scale in log-parameterized form, the inverse of `to_linear_scales`."""
        if self.scale_activation == "log":
            return self.scales
        return np.log(np.clip(self.scales, 1e-12, None))

    def to_logit_opacities(self) -> np.ndarray:
        """Opacity in logit (pre-sigmoid) form, the inverse of `to_activated_opacities`."""
        if self.opacity_activation == "logit":
            return self.opacities
        p = np.clip(self.opacities, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))
