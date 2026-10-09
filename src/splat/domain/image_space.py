"""Value objects for the 2D-image bounded contexts (generation, segmentation,
depth) that sit upstream of Gaussian reconstruction — a photo or a generated
image is not yet a GaussianCloud, but these are the pieces a future
`lift_to_gaussians(sticker, depth_map)` domain service would combine."""

from dataclasses import dataclass, field

import numpy as np

from splat.domain.errors import SplatDomainError


@dataclass
class Sticker:
    """A single segmented subject cut out of an image: an RGBA crop plus the
    mask's provenance in the source image."""

    rgba: np.ndarray  # (H, W, 4) uint8, straight (non-premultiplied) alpha
    bbox: tuple[int, int, int, int]  # (x0, y0, w, h) in the source image
    score: float  # predicted IoU / confidence from the segmentation model
    area: int

    @classmethod
    def from_mask(cls, image: np.ndarray, mask: np.ndarray, score: float) -> "Sticker":
        """A tight RGBA cutout of `image` where `mask` (H, W bool) is set."""
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            raise SplatDomainError("The prompt produced an empty mask.")
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        rgba = np.dstack([image[y0:y1, x0:x1, :3], mask[y0:y1, x0:x1] * 255]).astype(np.uint8)
        return cls(rgba, (x0, y0, x1 - x0, y1 - y0), score, int(mask.sum()))


@dataclass
class DepthMap:
    """Per-pixel depth aligned to the source image's resolution: metres, or
    relative disparity (higher = closer) when `units == "disparity"`."""

    depth: np.ndarray  # (H, W) float32
    focal_length_px: float | None = None
    field_of_view_deg: float | None = None
    units: str = "metres"
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class Shape3D:
    """A predicted 3D mesh: vertex positions and triangle indices, plus an
    optional per-vertex UV + source texture for the surface it was lifted
    from. Gaussian splats use `ManifestKind.GAUSSIAN_CLOUD` instead."""

    vertices: np.ndarray  # (V, 3) float32, meters
    faces: np.ndarray  # (F, 3) int64, indices into vertices
    uv: np.ndarray | None = None  # (V, 2) float32, [0, 1]
    texture: np.ndarray | None = None  # (H, W, 3) uint8 RGB
    colors: np.ndarray | None = None  # (V, 3) uint8 RGB per vertex
    metadata: dict[str, object] = field(default_factory=dict)
