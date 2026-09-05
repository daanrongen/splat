"""The universal currency that flows through the generative pipeline
(diffuse -> segment -> depth -> mesh -> gaussian), the same role
GaussianCloud plays for format conversion: one canonical shape every stage
consumes and produces, so stages chain without knowing about each other.

`ManifestKind` is deliberately flat (not a class hierarchy) — composable
capability tags in `KIND_TAGS` express "is-a" relationships (a sticker is a
colorlike raster, same as an image) without forcing every future kind into a
rigid subtype tree. `domain.contracts` builds on these tags to declare what
each stage accepts.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class ManifestKind(StrEnum):
    IMAGE = "image"  # generic RGB raster
    STICKER = "sticker"  # RGBA cutout + mask metadata — "is-a" image
    CAPTION = "caption"  # UTF-8 image-to-text output (.txt)
    EMBEDDING = "embedding"  # normalized image/text vector (.npy)
    DEPTH_MAP = "depth_map"  # per-pixel metric depth, cached losslessly (.npy)
    SHAPE_3D = "shape_3d"  # mesh or point cloud
    GAUSSIAN_CLOUD = "gaussian_cloud"  # canonical GaussianCloud cached as lossless .ply


KIND_TAGS: dict[ManifestKind, frozenset[str]] = {
    ManifestKind.IMAGE: frozenset({"raster", "rgb", "colorlike"}),
    ManifestKind.STICKER: frozenset({"raster", "rgba", "colorlike"}),
    ManifestKind.CAPTION: frozenset({"text"}),
    ManifestKind.EMBEDDING: frozenset({"vector"}),
    ManifestKind.DEPTH_MAP: frozenset({"raster", "single_channel", "metric"}),
    ManifestKind.SHAPE_3D: frozenset({"mesh_3d"}),
    ManifestKind.GAUSSIAN_CLOUD: frozenset({"splat_3d"}),
}


@dataclass
class Manifest:
    """A cached, content-addressed artifact. `id` is a deterministic hash of
    the inputs that produced it (stage, model, params, parents) — the same
    inputs always resolve to the same id, giving free memoization."""

    id: str
    kind: ManifestKind
    content_path: Path
    metadata: "ManifestMetadata"  # noqa: F821 — typed per kind, see domain/manifest_metadata.py
    params: dict = field(default_factory=dict)  # stage-invocation args (prompt, steps, seed, ...)
    parent_ids: list[str] = field(default_factory=list)
    created_by: str = ""  # "<stage>:<model-name>"
    content_size: int = 0  # bytes, of content_path
    content_sha256: str = ""
    created_at: str = ""  # ISO 8601 UTC, set once at first `put`
