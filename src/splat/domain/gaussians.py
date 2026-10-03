"""The canonical Gaussian aggregate — the hub every format adapter and every
reconstruction/compression adapter reads and writes, instead of N^2 pairwise
converters.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

from splat.domain.errors import InvalidGaussianCloud, UnsupportedSHDegree
from splat.domain.value_objects import ModelLicense, convention_flip_matrix

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
    metric_scale: bool = False  # True when `means` are in real-world units
    point_count: int = 0
    sh_degree: int = 0
    scale_activation: ScaleActivation = "log"
    opacity_activation: OpacityActivation = "logit"
    capture_camera_position: list[float] | None = None  # (3,) world-space, same frame as `means`
    capture_camera_rotation: list[list[float]] | None = None  # (3,3) world-to-camera, COLMAP/OpenCV
    capture_camera_intrinsics: list[float] | None = None  # [fx, fy, cx, cy, width, height]
    capture_camera_count: int | None = None  # total cameras the reconstruction backend registered
    # Every registered camera as {position, rotation, intrinsics, input}, same conventions as above.
    source_cameras: list[dict] | None = None
    quality: dict | None = None  # see domain/quality.py


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

        self.metadata.point_count = n
        self.metadata.sh_degree = self.sh_degree
        self.metadata.scale_activation = self.scale_activation
        self.metadata.opacity_activation = self.opacity_activation

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


def to_convention(cloud: GaussianCloud, target: str) -> GaussianCloud:
    """Re-express a cloud in another axis convention, metadata included.

    Only colmap <-> opengl is needed, and both directions are the same proper
    180-degree rotation about X, so one branch covers them. Everything that has
    an orientation moves together: point positions, per-Gaussian rotations, and
    the captured camera pose. Getting one of those wrong is exactly how a cloud
    ends up upside down or with the camera facing away from the scene.
    """
    source = cloud.metadata.coordinate_convention
    if source == target:
        return cloud

    flip = convention_flip_matrix(source, target)
    metadata = replace(
        cloud.metadata,
        coordinate_convention=target,
        # colmap is +Y down, so "y" is only an honest answer for opengl.
        up_axis="y" if target == "opengl" else cloud.metadata.up_axis,
    )
    if metadata.capture_camera_position is not None:
        metadata.capture_camera_position = _flip_position(metadata.capture_camera_position, flip)
    if metadata.capture_camera_rotation is not None:
        metadata.capture_camera_rotation = _flip_rotation(metadata.capture_camera_rotation, flip)
    if metadata.source_cameras is not None:
        metadata.source_cameras = [
            {
                **camera,
                "position": _flip_position(camera["position"], flip),
                "rotation": _flip_rotation(camera["rotation"], flip),
            }
            for camera in metadata.source_cameras
        ]

    return replace(
        cloud,
        means=(cloud.means @ flip).astype(np.float32),
        rotations=_flip_quaternions(cloud.rotations, flip),
        metadata=metadata,
    )


def _flip_position(position: list[float], flip: np.ndarray) -> list[float]:
    return (np.asarray(position, dtype=np.float32) @ flip).tolist()


def _flip_rotation(rotation: list[list[float]], flip: np.ndarray) -> list[list[float]]:
    # Stored world-to-camera. The camera's own local axes differ by the same
    # flip (colmap looks down +Z with +Y down, opengl down -Z with +Y up),
    # so it conjugates on both sides of the camera-to-world matrix.
    return (flip @ np.asarray(rotation, dtype=np.float32).T @ flip).T.tolist()


def _flip_quaternions(quats: np.ndarray, flip: np.ndarray) -> np.ndarray:
    """Conjugating a (w,x,y,z) quaternion by a 180-degree axis flip negates the
    two components whose axes flipped - derived from the Hamilton product, and
    checked against a matrix round-trip in the tests."""
    signs = np.ones(4, dtype=np.float32)
    signs[1:] = np.diag(flip)
    return (quats * signs).astype(np.float32)


def normalize_gaussian_cloud(cloud: GaussianCloud, *, target_radius: float = 1.0) -> GaussianCloud:
    """Recenter on the robust (median) centroid and rescale so the median
    distance from center is `target_radius`.

    Feed-forward and SfM-based reconstruction have no absolute scale or
    canonical origin — this makes output consistent capture-to-capture
    instead of carrying over whatever arbitrary position/scale the
    backend's world frame happened to produce. Rotations, opacities, and SH
    coefficients are untouched: recentering is a pure translation and
    rescaling is uniform, so orientation and color are unaffected. This does
    not change axis convention (see `GaussianCloudMetadata.coordinate_convention`
    for that) — median is used over mean so a handful of far-flung outlier
    points (common in sparse SfM reconstructions) don't skew the center/scale.
    """
    center = np.median(cloud.means, axis=0)
    means = cloud.means - center

    radius = float(np.median(np.linalg.norm(means, axis=1)))
    scale_factor = target_radius / radius if radius > 1e-8 else 1.0
    means = (means * scale_factor).astype(np.float32)

    linear_scales = cloud.to_linear_scales() * scale_factor
    if cloud.scale_activation == "log":
        scales = np.log(np.clip(linear_scales, 1e-12, None)).astype(np.float32)
    else:
        scales = linear_scales.astype(np.float32)

    # Camera positions live in the same world frame as `means`, so they get the same
    # recenter+rescale; rotations are untouched by pure translation + uniform scale.
    def move(position: list[float]) -> list[float]:
        return ((np.asarray(position, dtype=np.float32) - center) * scale_factor).tolist()

    metadata = cloud.metadata
    if metadata.capture_camera_position is not None:
        metadata = replace(metadata, capture_camera_position=move(metadata.capture_camera_position))
    if metadata.source_cameras is not None:
        cameras = [{**c, "position": move(c["position"])} for c in metadata.source_cameras]
        metadata = replace(metadata, source_cameras=cameras)

    return replace(cloud, means=means, scales=scales, metadata=metadata)


def with_view(cloud: GaussianCloud, index: int) -> GaussianCloud:
    """The cloud with source camera `index` as its capture pose, for rendering an input view."""
    cameras = cloud.metadata.source_cameras or []
    if not 0 <= index < len(cameras):
        raise InvalidGaussianCloud(
            f"View {index} does not exist; this cloud has {len(cameras)} source camera(s)."
        )
    camera = cameras[index]
    metadata = replace(
        cloud.metadata,
        capture_camera_position=camera["position"],
        capture_camera_rotation=camera["rotation"],
        capture_camera_intrinsics=camera["intrinsics"],
    )
    return replace(cloud, metadata=metadata)
