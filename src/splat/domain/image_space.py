"""Value objects for the 2D-image bounded contexts (generation, segmentation,
depth) that sit upstream of Gaussian reconstruction — a photo or a generated
image is not yet a GaussianCloud, but these are the pieces a future
`lift_to_gaussians(sticker, depth_map)` domain service would combine."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Sticker:
    """A single segmented subject cut out of an image: an RGBA crop plus the
    mask's provenance in the source image."""

    rgba: np.ndarray  # (H, W, 4) uint8, straight (non-premultiplied) alpha
    bbox: tuple[int, int, int, int]  # (x0, y0, w, h) in the source image
    score: float  # predicted IoU / confidence from the segmentation model
    area: int


@dataclass
class DepthMap:
    """Per-pixel depth in meters, aligned to the source image's resolution."""

    depth: np.ndarray  # (H, W) float32, meters
    focal_length_px: float | None = None
    field_of_view_deg: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class Shape3D:
    """A predicted 3D mesh: vertex positions and triangle indices, plus an
    optional per-vertex UV + source texture for the surface it was lifted
    from. Gaussian splats use `AssetKind.GAUSSIAN_CLOUD` instead."""

    vertices: np.ndarray  # (V, 3) float32, meters
    faces: np.ndarray  # (F, 3) int64, indices into vertices
    uv: np.ndarray | None = None  # (V, 2) float32, [0, 1]
    texture: np.ndarray | None = None  # (H, W, 3) uint8 RGB
    metadata: dict[str, object] = field(default_factory=dict)
