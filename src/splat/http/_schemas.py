"""Pydantic wire models for the HTTP driving adapter. Kept separate from
handlers/*.py's plain dataclasses so that layer stays framework-agnostic —
these are the one place "what a request/response looks like on the wire"
is defined, reused by RemoteSplatClient (SPLAT_URL) later.
"""

from pydantic import BaseModel

from splat.domain.asset import Asset


class DiffuseBody(BaseModel):
    prompt: str
    model: str = "sdxl-turbo-mlx"
    negative_prompt: str = ""
    steps: int | None = None
    seed: int | None = None
    device: str = "auto"


class AssetSummary(BaseModel):
    id: str
    kind: str
    metadata: dict
    parent_ids: list[str]
    created_by: str

    @classmethod
    def from_asset(cls, asset: Asset) -> "AssetSummary":
        return cls(
            id=asset.id,
            kind=asset.kind.value,
            metadata=asset.metadata,
            parent_ids=asset.parent_ids,
            created_by=asset.created_by,
        )


class ModelSummary(BaseModel):
    name: str
    runtime: str
    license: str
    cached: bool | None = None


class ModelInfoResponse(BaseModel):
    name: str
    runtime: str
    source: str
    license: str
    min_images: int | None = None
    max_images: int | None = None


class InfoResponse(BaseModel):
    format: str | None
    points: int
    sh_degree: int
    bbox_min: list[float]
    bbox_max: list[float]


class ValidationResponse(BaseModel):
    valid: bool
    issues: list[str]
    points: int
