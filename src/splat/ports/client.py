"""The seam between a command and "where the work actually runs":
LocalSplatClient (adapters/client/local.py) calls handlers/*.py in-process;
RemoteSplatClient (adapters/client/http.py) calls a remote `splat http`
server instead. Every CLI command goes through registry.wiring.get_client()
rather than importing handlers/*.py directly, so SPLAT_URL transparently
redirects execution without any command needing to know which client it got.

Asset-producing methods return real domain objects (Asset, GaussianCloud)
because their payload — file bytes — is fully transmitted and either
lands in the local asset cache (asset-producing) or gets written to a
caller-supplied local path and re-read (gaussian/tools convert/tools compress).
info/validate/models_* return plain summaries instead of full domain
objects, because http/'s wire schema for those is intentionally a summary,
not a full GaussianCloud/catalog-descriptor — this is the "presentation of
a read" boundary, not something a remote round-trip can reconstruct in
full, so both clients report through the same reduced shape.

`splat tools displace.height` isn't part of this contract: `splat http`
doesn't expose a route for it (out of #13's scope), so it stays wired
directly to handlers.tools.displace_height, unaffected by SPLAT_URL, until
a future PR adds that route.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from splat.application.tools.convert import ConvertResult
from splat.domain.asset import Asset
from splat.domain.gaussians import GaussianCloud
from splat.handlers.depth import DepthRequest
from splat.handlers.diffuse import DiffuseRequest, DiffuseResult
from splat.handlers.gaussian import GaussianRequest, GaussianResult
from splat.handlers.mesh import MeshRequest
from splat.handlers.segment import SegmentRequest
from splat.handlers.tools.compress import CompressRequest
from splat.handlers.tools.convert import ConvertRequest


@dataclass(frozen=True)
class InfoSummary:
    format: str | None
    points: int
    sh_degree: int
    bbox_min: list[float]
    bbox_max: list[float]


@dataclass(frozen=True)
class ValidationSummary:
    valid: bool
    issues: list[str]
    points: int


@dataclass(frozen=True)
class ModelSummary:
    name: str
    runtime: str
    license: str
    cached: bool | None = None


@dataclass(frozen=True)
class ModelInfo:
    name: str
    runtime: str
    source: str
    license: str
    min_images: int | None = None
    max_images: int | None = None


class SplatClient(Protocol):
    def diffuse(self, request: DiffuseRequest) -> DiffuseResult: ...

    def segment(self, request: SegmentRequest) -> list[Asset]: ...

    def depth(self, request: DepthRequest) -> list[Asset]: ...

    def mesh(self, request: MeshRequest) -> list[Asset]: ...

    def gaussian(self, request: GaussianRequest) -> GaussianResult: ...

    def convert(self, request: ConvertRequest) -> ConvertResult: ...

    def compress(self, request: CompressRequest) -> GaussianCloud: ...

    def info(self, path: Path) -> InfoSummary: ...

    def validate(self, path: Path, *, strict: bool = False) -> ValidationSummary: ...

    def models_list(self) -> list[ModelSummary]: ...

    def models_pull(self, name: str) -> None: ...

    def models_info(self, name: str) -> ModelInfo: ...

    def models_rm(self, name: str) -> None: ...
