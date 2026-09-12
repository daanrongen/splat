"""Pydantic wire models for the HTTP driving adapter. Kept separate from
handlers/*.py's plain dataclasses so that layer stays framework-agnostic —
these are the one place "what a request/response looks like on the wire"
is defined, reused by RemoteSplatClient (SPLAT_URL) later.
"""

import dataclasses

from pydantic import BaseModel

from splat.domain.manifest import Manifest


class DiffuseBody(BaseModel):
    prompt: str
    model: str = "sdxl-turbo-mlx"
    negative_prompt: str = ""
    steps: int | None = None
    strength: float | None = None
    seed: int | None = None
    device: str = "auto"


class UpscaleBody(BaseModel):
    model: str = "realesrgan-mlx"
    factor: int = 4
    tile: int = 0


class EmbedBody(BaseModel):
    text: str | None = None
    model: str = "mobileclip2-s0"
    device: str = "auto"


class ManifestSummary(BaseModel):
    id: str
    kind: str
    created_by: str
    created_at: str
    content_size: int
    content_sha256: str
    parent_ids: list[str]

    @classmethod
    def from_manifest(cls, manifest: Manifest) -> "ManifestSummary":
        return cls(
            id=manifest.id,
            kind=manifest.kind.value,
            created_by=manifest.created_by,
            created_at=manifest.created_at,
            content_size=manifest.content_size,
            content_sha256=manifest.content_sha256,
            parent_ids=manifest.parent_ids,
        )


class ManifestDetail(ManifestSummary):
    metadata: dict
    params: dict

    @classmethod
    def from_manifest(cls, manifest: Manifest) -> "ManifestDetail":
        return cls(
            **ManifestSummary.from_manifest(manifest).model_dump(),
            metadata=dataclasses.asdict(manifest.metadata),
            params=manifest.params,
        )


class ModelSummary(BaseModel):
    name: str
    stage: str
    runtime: str
    license: str
    commercial: bool = True
    cached: bool | None = None


class ModelInfoResponse(BaseModel):
    name: str
    runtime: str
    source: str
    license: str
    min_images: int | None = None
    max_images: int | None = None
    dimension: int | None = None
    normalized: bool | None = None
    notes: str = ""


class InfoResponse(BaseModel):
    format: str | None
    points: int
    sh_degree: int
    bbox_min: list[float]
    bbox_max: list[float]
    coordinate_convention: str
    up_axis: str
    source_model: str | None
    license: str | None
    capture_camera_count: int | None


class ValidationResponse(BaseModel):
    valid: bool
    issues: list[str]
    points: int
