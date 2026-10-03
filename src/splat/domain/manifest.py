"""The universal currency that flows through the generative pipeline
(diffuse -> segment -> depth -> gaussian -> render), the same role
GaussianCloud plays for format conversion: one canonical shape every stage
consumes and produces, so stages chain without knowing about each other.

`ManifestKind` is deliberately flat (not a class hierarchy) — composable
capability tags in `KIND_TAGS` express "is-a" relationships (a sticker is a
colorlike raster, same as an image) without forcing every future kind into a
rigid subtype tree. `domain.contracts` builds on these tags to declare what
each stage accepts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from splat.domain.gaussians import GaussianCloud
    from splat.domain.manifest_metadata import ManifestMetadata


class ManifestKind(StrEnum):
    IMAGE = "image"  # generic RGB raster
    STICKER = "sticker"  # RGBA cutout + mask metadata — "is-a" image
    CAPTION = "caption"  # UTF-8 image-to-text output (.txt)
    EMBEDDING = "embedding"  # normalized image/text vector (.npy)
    DEPTH_MAP = "depth_map"  # per-pixel metric depth, cached losslessly (.npy)
    SHAPE_3D = "shape_3d"  # mesh or point cloud
    GAUSSIAN_CLOUD = "gaussian_cloud"  # canonical GaussianCloud cached as lossless .ply
    FAN_OUT = "fan_out"  # empty marker listing the children of a fan-out stage


KIND_TAGS: dict[ManifestKind, frozenset[str]] = {
    ManifestKind.IMAGE: frozenset({"raster", "rgb", "colorlike"}),
    ManifestKind.STICKER: frozenset({"raster", "rgba", "colorlike"}),
    ManifestKind.CAPTION: frozenset({"text"}),
    ManifestKind.EMBEDDING: frozenset({"vector"}),
    ManifestKind.DEPTH_MAP: frozenset({"raster", "single_channel"}),
    ManifestKind.SHAPE_3D: frozenset({"mesh_3d"}),
    ManifestKind.GAUSSIAN_CLOUD: frozenset({"splat_3d"}),
    ManifestKind.FAN_OUT: frozenset(),
}


@dataclass
class Manifest:
    """A cached artifact. For pipeline-produced manifests, `id` is a
    deterministic hash of the inputs that produced it (stage, model, params,
    parents) — the same inputs always resolve to the same id, giving free
    memoization. Manifests registered from an external file instead (no
    invocation to hash) are addressed by content: `id` is a hash of the bytes
    themselves, and a file whose bytes already exist under a pipeline-produced
    id resolves to that id rather than minting a second one — see
    `FilesystemManifestRepository.put_external`."""

    id: str
    kind: ManifestKind
    content_path: Path
    metadata: ManifestMetadata  # typed per kind, see domain/manifest_metadata.py
    params: dict = field(default_factory=dict)  # stage-invocation args (prompt, steps, seed, ...)
    parent_ids: list[str] = field(default_factory=list)
    created_by: str = ""  # "<stage>:<model-name>"
    content_size: int = 0  # bytes, of content_path
    content_sha256: str = ""
    created_at: str = ""  # ISO 8601 UTC, set once at first `put`
    label: str = ""  # user-given name, never part of the cache key

    @classmethod
    def load(cls, id: str) -> Manifest:
        """Fetch a cached manifest by id (accepts an optional leading `@`)."""
        from splat.registry.wiring import get_manifest_repository

        return get_manifest_repository().get(id.removeprefix("@"))

    def as_image(self) -> np.ndarray:
        """Decode this manifest's content as an RGB/RGBA array (IMAGE/STICKER kinds)."""
        from splat.adapters.formats.image import read_rgb_or_rgba

        return read_rgb_or_rgba(self.content_path)

    def as_text(self) -> str:
        """Read this manifest's content as UTF-8 text (CAPTION kind)."""
        return self.content_path.read_text(encoding="utf-8")

    def as_array(self) -> np.ndarray:
        """Load this manifest's content as a numpy array (EMBEDDING/DEPTH_MAP kinds)."""
        import numpy as np

        return np.load(self.content_path)

    def as_gaussian_cloud(self) -> GaussianCloud:
        """Parse this manifest's content as a GaussianCloud (GAUSSIAN_CLOUD kind)."""
        from splat.registry.wiring import get_reader

        return get_reader(self.content_path.suffix).read(self.content_path)
