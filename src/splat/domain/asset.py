"""The universal envelope that flows through the generative pipeline
(diffuse -> segment -> depth -> mesh -> gaussian), the same role
GaussianCloud plays for format conversion: one canonical shape every stage
consumes and produces, so stages chain without knowing about each other.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class AssetKind(StrEnum):
    IMAGE = "image"  # generic RGB raster
    STICKER = "sticker"  # RGBA cutout + mask metadata — "is-a" image
    CAPTION = "caption"  # UTF-8 image-to-text output (.txt)
    EMBEDDING = "embedding"  # normalized image/text vector (.npy)
    DEPTH_MAP = "depth_map"  # per-pixel metric depth, cached losslessly (.npy)
    SHAPE_3D = "shape_3d"  # mesh or point cloud, not yet Gaussians (future)
    GAUSSIAN_CLOUD = "gaussian_cloud"  # future: wraps domain.gaussians.GaussianCloud


@dataclass
class Asset:
    """A cached, content-addressed artifact. `id` is a deterministic hash of
    the inputs that produced it (stage, model, params, parents) — the same
    inputs always resolve to the same id, giving free memoization."""

    id: str
    kind: AssetKind
    content_path: Path
    metadata: dict = field(default_factory=dict)
    parent_ids: list[str] = field(default_factory=list)
    created_by: str = ""  # "<stage>:<model-name>"
