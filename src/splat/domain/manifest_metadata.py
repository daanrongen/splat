"""Typed per-kind metadata for `Manifest.metadata` — one closed dataclass
shape per `ManifestKind`, replacing the free-form dict every stage used to
stuff arbitrary keys into. Fields are optional where a kind has more than one
producer (e.g. IMAGE from both `diffuse` and `upscale`) rather than splitting
by (kind, stage) — one shape per kind keeps `KIND_METADATA_CLS` a simple
lookup instead of a second dispatch axis.
"""

import dataclasses
from dataclasses import dataclass, field

from splat.domain.gaussians import GaussianCloudMetadata
from splat.domain.manifest import ManifestKind
from splat.domain.value_objects import ModelLicense


@dataclass
class RasterMetadata:
    """IMAGE: diffuse leaves every field at its default; upscale fills them in."""

    variant: str | None = None
    source_width: int | None = None
    source_height: int | None = None
    output_width: int | None = None
    output_height: int | None = None


@dataclass
class StickerMetadata:
    bbox: tuple[int, int, int, int]
    score: float
    area: int
    width: int | None = None
    height: int | None = None


@dataclass
class SegmentManifestMetadata:
    """The empty `fan_out` marker `run_segment` caches at the parent key."""

    children: list[str]


@dataclass
class CaptionMetadata:
    text_length: int
    model: str | None = None


@dataclass
class EmbeddingMetadata:
    input_type: str
    dtype: str
    shape: list[int]
    dimension: int
    normalized: bool
    model: str
    text_sha256: str | None = None
    text_length: int | None = None


@dataclass
class DepthMetadata:
    focal_length_px: float | None = None
    field_of_view_deg: float | None = None
    width: int | None = None
    height: int | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class MeshMetadata:
    extra: dict = field(default_factory=dict)


ManifestMetadata = (
    RasterMetadata
    | StickerMetadata
    | SegmentManifestMetadata
    | CaptionMetadata
    | EmbeddingMetadata
    | DepthMetadata
    | MeshMetadata
    | GaussianCloudMetadata
)

KIND_METADATA_CLS: dict[ManifestKind, type] = {
    ManifestKind.IMAGE: RasterMetadata,
    ManifestKind.STICKER: StickerMetadata,
    ManifestKind.CAPTION: CaptionMetadata,
    ManifestKind.EMBEDDING: EmbeddingMetadata,
    ManifestKind.DEPTH_MAP: DepthMetadata,
    ManifestKind.SHAPE_3D: MeshMetadata,
    ManifestKind.GAUSSIAN_CLOUD: GaussianCloudMetadata,
    ManifestKind.FAN_OUT: SegmentManifestMetadata,
}


def metadata_to_dict(metadata: ManifestMetadata) -> dict:
    return dataclasses.asdict(metadata)


def metadata_from_dict(kind: ManifestKind, data: dict) -> ManifestMetadata:
    cls = KIND_METADATA_CLS[kind]
    if cls is GaussianCloudMetadata and data.get("license") is not None:
        data = {**data, "license": ModelLicense(**data["license"])}
    return cls(**data)
