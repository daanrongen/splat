"""The seam between a command and where the work actually runs.

LocalSplatClient calls handlers in-process. RemoteSplatClient calls a remote
`splat http` server. CLI commands use registry.wiring.get_client(), so
SPLAT_URL redirects remote-capable execution without changing command code.

Manifest-producing methods return real Manifest objects because their file bytes are
transmitted and land in the local asset cache. Read-only
methods return summaries because the HTTP wire schema intentionally exposes
presentation data, not every internal domain detail.

`splat render`, `splat mesh` and `splat export` aren't part of this contract:
they run where their inputs and tools (Blender, Open3D, the -o path) live, so
they call their handlers directly, unaffected by SPLAT_URL.

`splat models prune` is deliberately not on this contract either: it deletes
weights from whichever machine's disk it runs on, and a route that lets a
client reclaim a shared server's model cache is a footgun, not parity.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from splat.domain.manifest import Manifest
from splat.handlers.caption import CaptionRequest
from splat.handlers.depth import DepthRequest
from splat.handlers.diffuse import DiffuseRequest, DiffuseResult
from splat.handlers.embed import EmbedRequest
from splat.handlers.gaussian import GaussianRequest
from splat.handlers.segment import SegmentRequest
from splat.handlers.upscale import UpscaleRequest


@dataclass(frozen=True)
class InfoSummary:
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


@dataclass(frozen=True)
class ValidationSummary:
    valid: bool
    issues: list[str]
    points: int


@dataclass(frozen=True)
class ModelSummary:
    name: str
    stage: str
    runtime: str
    license: str  # SPDX id only; see `commercial` for the use restriction
    commercial: bool = True
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

    def info(self, path: Path) -> InfoSummary: ...

    def validate(self, path: Path, *, strict: bool = False) -> ValidationSummary: ...

    def models_list(self) -> list[ModelSummary]: ...

    def models_pull(self, name: str) -> None: ...

    def models_info(self, name: str) -> ModelInfo: ...

    def models_rm(self, name: str) -> None: ...
