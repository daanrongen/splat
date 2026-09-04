"""The seam between a command and where the work actually runs.

LocalSplatClient calls handlers in-process. RemoteSplatClient calls a remote
`splat http` server. CLI commands use registry.wiring.get_client(), so
SPLAT_URL redirects remote-capable execution without changing command code.

Asset-producing methods return real domain objects (Asset, GaussianCloud)
because their file bytes are transmitted and either land in the local asset
cache or get written to a caller-supplied local path and re-read. Read-only
methods return summaries because the HTTP wire schema intentionally exposes
presentation data, not every internal domain detail.

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
from splat.handlers.caption import CaptionRequest
from splat.handlers.depth import DepthRequest
from splat.handlers.diffuse import DiffuseRequest, DiffuseResult
from splat.handlers.embed import EmbedRequest
from splat.handlers.gaussian import GaussianRequest, GaussianResult
from splat.handlers.mesh import MeshRequest
from splat.handlers.segment import SegmentRequest
from splat.handlers.tools.compress import CompressRequest
from splat.handlers.tools.convert import ConvertRequest
from splat.handlers.upscale import UpscaleRequest


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
    dimension: int | None = None
    normalized: bool | None = None
    notes: str = ""


class SplatClient(Protocol):
    def diffuse(self, request: DiffuseRequest) -> DiffuseResult: ...

    def segment(self, request: SegmentRequest) -> list[Asset]: ...

    def caption(self, request: CaptionRequest) -> list[Asset]: ...

    def depth(self, request: DepthRequest) -> list[Asset]: ...

    def upscale(self, request: UpscaleRequest) -> list[Asset]: ...

    def embed(self, request: EmbedRequest) -> list[Asset]: ...

    def mesh(self, request: MeshRequest) -> list[Asset]: ...

    def gaussian(self, request: GaussianRequest) -> GaussianResult: ...

    def tools_convert(self, request: ConvertRequest) -> ConvertResult: ...

    def tools_compress(self, request: CompressRequest) -> GaussianCloud: ...

    def info(self, path: Path) -> InfoSummary: ...

    def validate(self, path: Path, *, strict: bool = False) -> ValidationSummary: ...

    def models_list(self) -> list[ModelSummary]: ...

    def models_pull(self, name: str) -> None: ...

    def models_info(self, name: str) -> ModelInfo: ...

    def models_rm(self, name: str) -> None: ...
