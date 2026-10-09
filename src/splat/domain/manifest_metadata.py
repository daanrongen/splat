"""Typed per-kind metadata for `Manifest.metadata` — one closed dataclass
shape per `ManifestKind`, replacing the free-form dict every stage used to
stuff arbitrary keys into. Fields are optional where a kind has more than one
producer (e.g. IMAGE from both `diffuse` and `upscale`) rather than splitting
by (kind, stage) — one shape per kind keeps `KIND_METADATA_CLS` a simple
lookup instead of a second dispatch axis.
"""

import dataclasses
import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from io import BytesIO

import numpy as np

from splat.domain.errors import ContractViolation
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
    units: str = "metres"  # "metres" (near = low) or "disparity" (near = high, relative)
    extra: dict = field(default_factory=dict)


@dataclass
class MeshMetadata:
    vertex_count: int = 0
    face_count: int = 0
    textured: bool = False
    resolution: int | None = None  # voxels along the longest axis


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
    if cls is DepthMetadata and "units" not in data and data.get("extra", {}).get("relative"):
        data = {**data, "units": "disparity"}  # maps cached before `units` existed
    if cls is MeshMetadata and "extra" in data:
        data = data["extra"]  # meshes cached before typed fields
    if cls is GaussianCloudMetadata and data.get("license") is not None:
        data = {**data, "license": ModelLicense(**data["license"])}
    return cls(**data)


_PNG = b"\x89PNG\r\n\x1a\n"
_PNG_RGB, _PNG_RGBA = 2, 6


def _png(content: bytes) -> tuple[int, int, int] | None:
    """(width, height, colour type) from the IHDR chunk, or None for a non-PNG."""
    if not content.startswith(_PNG) or len(content) < 26:
        return None
    width, height = struct.unpack(">II", content[16:24])
    return width, height, content[25]


def _npy(content: bytes) -> tuple[tuple[int, ...], np.dtype] | None:
    try:
        buf = BytesIO(content)
        major, _ = np.lib.format.read_magic(buf)
        read_header = (
            np.lib.format.read_array_header_1_0
            if major == 1
            else np.lib.format.read_array_header_2_0
        )
        shape, _, dtype = read_header(buf)
    except ValueError:
        return None
    return shape, dtype


def _ply_vertex_count(content: bytes) -> int | None:
    header = content[: content.find(b"end_header")].decode("ascii", "replace")
    if not header.startswith("ply"):
        return None
    for line in header.splitlines():
        if line.startswith("element vertex "):
            return int(line.split()[2])
    return None


def _check_image(m: RasterMetadata, content: bytes) -> str | None:
    png = _png(content)
    if png is None or png[2] not in (_PNG_RGB, _PNG_RGBA):
        return "content must be an RGB or RGBA PNG"
    if m.output_width is not None and (m.output_width, m.output_height) != png[:2]:
        return f"metadata says {m.output_width}x{m.output_height}, PNG is {png[0]}x{png[1]}"
    return None


def _check_sticker(m: StickerMetadata, content: bytes) -> str | None:
    png = _png(content)
    if png is None or png[2] != _PNG_RGBA:
        return "content must be an RGBA PNG"
    if len(m.bbox) != 4 or m.bbox[2] <= 0 or m.bbox[3] <= 0:
        return f"bbox must be (x, y, width, height) with a positive size, got {m.bbox}"
    return None


def _check_caption(m: CaptionMetadata, content: bytes) -> str | None:
    text = content.decode("utf-8", "replace")
    if not text.strip():
        return "caption is empty"
    return None if m.text_length == len(text) else "text_length does not match the text"


def _check_embedding(m: EmbeddingMetadata, content: bytes) -> str | None:
    npy = _npy(content)
    if npy is None or list(npy[0]) != m.shape or str(npy[1]) != m.dtype:
        return f"content must be a {m.dtype} .npy of shape {m.shape}"
    return None if m.dimension == m.shape[-1] else "dimension must be the last axis size"


def _check_depth(m: DepthMetadata, content: bytes) -> str | None:
    npy = _npy(content)
    if npy is None or len(npy[0]) != 2 or npy[1] != np.float32:
        return "content must be a 2D float32 .npy"
    if m.units not in ("metres", "disparity"):
        return f"units must be metres or disparity, got {m.units!r}"
    if m.width is not None and (m.height, m.width) != npy[0]:
        return f"metadata says {m.width}x{m.height}, array is {npy[0][1]}x{npy[0][0]}"
    return None


def _check_gaussian_cloud(m: GaussianCloudMetadata, content: bytes) -> str | None:
    if m.coordinate_convention != "opengl":
        return f"must be in the opengl convention, got {m.coordinate_convention!r}"
    if not 0 <= m.sh_degree <= 3:
        return f"sh_degree must be 0 to 3, got {m.sh_degree}"
    count = _ply_vertex_count(content)
    if count is None or count != m.point_count or count == 0:
        return f"content must be a .ply with point_count={m.point_count} Gaussians"
    return None


def _check_mesh(m: MeshMetadata, content: bytes) -> str | None:
    return None if m.vertex_count and m.face_count else "mesh has no vertices or faces"


_CONTRACTS: dict[ManifestKind, Callable] = {
    ManifestKind.IMAGE: _check_image,
    ManifestKind.STICKER: _check_sticker,
    ManifestKind.CAPTION: _check_caption,
    ManifestKind.EMBEDDING: _check_embedding,
    ManifestKind.DEPTH_MAP: _check_depth,
    ManifestKind.GAUSSIAN_CLOUD: _check_gaussian_cloud,
    ManifestKind.SHAPE_3D: _check_mesh,
}


def check_contract(kind: ManifestKind, metadata: ManifestMetadata, content: bytes) -> None:
    """Raises ContractViolation when a stage output is not its kind's canonical form."""
    if not isinstance(metadata, KIND_METADATA_CLS[kind]):
        raise ContractViolation(f"{kind.value} needs {KIND_METADATA_CLS[kind].__name__}")
    check = _CONTRACTS.get(kind)
    if check is not None and (problem := check(metadata, content)) is not None:
        raise ContractViolation(f"{kind.value}: {problem}")
