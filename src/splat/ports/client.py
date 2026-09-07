"""The seam between a command and where the work actually runs.

LocalSplatClient calls handlers in-process. RemoteSplatClient calls a remote
`splat http` server. CLI commands use registry.wiring.get_client(), so
SPLAT_URL redirects remote-capable execution without changing command code.

Manifest-producing methods return real Manifest objects because their file bytes are
transmitted and land in the local asset cache. Read-only
methods return summaries because the HTTP wire schema intentionally exposes
presentation data, not every internal domain detail.

`splat tools displace.height` isn't part of this contract: `splat http`
doesn't expose a route for it (out of #13's scope), so it stays wired
directly to handlers.tools.displace_height, unaffected by SPLAT_URL, until
a future PR adds that route. `splat render` follows the same pattern (see
#50): its `blender` model needs a Blender install on whichever machine
actually renders, so it stays wired directly to handlers.render for now
rather than assuming SPLAT_URL's remote-execution model fits it unexamined.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from splat.application.tools.convert import ConvertResult
from splat.application.tools.extract_surface import ExtractSurfaceResult
from splat.domain.gaussians import GaussianCloud
from splat.domain.manifest import Manifest
from splat.handlers.caption import CaptionRequest
from splat.handlers.depth import DepthRequest
from splat.handlers.diffuse import DiffuseRequest, DiffuseResult
from splat.handlers.embed import EmbedRequest
from splat.handlers.gaussian import GaussianRequest
from splat.handlers.segment import SegmentRequest
from splat.handlers.tools.compress import CompressRequest
from splat.handlers.tools.convert import ConvertRequest
from splat.handlers.tools.declutter import DeclutterRequest
from splat.handlers.tools.extract_surface import ExtractSurfaceRequest
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

    def segment(self, request: SegmentRequest) -> list[Manifest]: ...

    def caption(self, request: CaptionRequest) -> list[Manifest]: ...

    def depth(self, request: DepthRequest) -> list[Manifest]: ...

    def upscale(self, request: UpscaleRequest) -> list[Manifest]: ...

    def embed(self, request: EmbedRequest) -> list[Manifest]: ...

    def gaussian(self, request: GaussianRequest) -> list[Manifest]: ...

    def tools_convert(self, request: ConvertRequest) -> ConvertResult: ...

    def tools_compress(self, request: CompressRequest) -> GaussianCloud: ...

    def tools_declutter(self, request: DeclutterRequest) -> GaussianCloud: ...

    def tools_extract_surface(self, request: ExtractSurfaceRequest) -> ExtractSurfaceResult: ...

    def info(self, path: Path) -> InfoSummary: ...

    def validate(self, path: Path, *, strict: bool = False) -> ValidationSummary: ...

    def models_list(self) -> list[ModelSummary]: ...

    def models_pull(self, name: str) -> None: ...

    def models_info(self, name: str) -> ModelInfo: ...

    def models_rm(self, name: str) -> None: ...
