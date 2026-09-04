"""HTTP client for a remote `splat http` server — used by the CLI when
SPLAT_URL is set. Reuses http/_schemas.py's pydantic models as the wire
contract, so there is exactly one definition of each request/response
shape, shared with the server that produces/consumes it.
"""

from pathlib import Path

import httpx

from splat.application.tools.convert import ConvertResult
from splat.domain.asset import Asset, AssetKind
from splat.domain.errors import SplatDomainError
from splat.domain.gaussians import GaussianCloud
from splat.handlers.depth import DepthRequest
from splat.handlers.diffuse import DiffuseRequest, DiffuseResult
from splat.handlers.gaussian import GaussianRequest, GaussianResult
from splat.handlers.mesh import MeshRequest
from splat.handlers.segment import SegmentRequest
from splat.handlers.tools.compress import CompressRequest
from splat.handlers.tools.convert import ConvertRequest
from splat.http._schemas import (
    AssetSummary,
    DiffuseBody,
    InfoResponse,
    ModelInfoResponse,
    ValidationResponse,
)
from splat.http._schemas import ModelSummary as WireModelSummary
from splat.ports.client import InfoSummary, ModelInfo, ModelSummary, ValidationSummary
from splat.registry.wiring import get_asset_cache, get_reader


def _raise_for_domain_error(response: httpx.Response) -> None:
    if response.status_code == 422:
        detail = response.json().get("detail", response.text)
        raise SplatDomainError(detail)
    response.raise_for_status()


class RemoteSplatClient:
    """Proxies every SplatClient method to a remote `splat http` server,
    storing asset-shaped results in the local asset cache under the same
    id the server reports, and writing file-shaped results to the
    caller-supplied local output path — so downstream CLI code (-o
    writing, NDJSON piping) works identically to LocalSplatClient."""

    def __init__(self, base_url: str, *, timeout: float = 300.0) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def _store_asset(
        self,
        asset_id: str,
        *,
        kind: AssetKind,
        content: bytes,
        ext: str,
        metadata: dict,
        parent_ids: list[str],
        created_by: str,
    ) -> Asset:
        cache = get_asset_cache()
        existing = cache.find(asset_id)
        if existing is not None:
            return existing
        return cache.put(
            asset_id,
            kind=kind,
            content_bytes=content,
            ext=ext,
            metadata=metadata,
            parent_ids=parent_ids,
            created_by=created_by,
        )

    def diffuse(self, request: DiffuseRequest) -> DiffuseResult:
        body = DiffuseBody(
            prompt=request.prompt,
            model=request.model,
            negative_prompt=request.negative_prompt,
            steps=request.steps,
            seed=request.seed,
            device=request.device,
        )
        response = self._client.post("/diffuse", json=body.model_dump())
        _raise_for_domain_error(response)
        asset = self._store_asset(
            response.headers["X-Splat-Asset-Id"],
            kind=AssetKind.IMAGE,
            content=response.content,
            ext="png",
            metadata={"prompt": request.prompt},
            parent_ids=[],
            created_by=f"diffuse:{request.model}",
        )
        return DiffuseResult(
            asset=asset, license_warning=response.headers.get("X-Splat-License-Warning")
        )

    def segment(self, request: SegmentRequest) -> list[Asset]:
        results: list[Asset] = []
        for input_asset in request.inputs:
            files = {
                "image": (input_asset.content_path.name, input_asset.content_path.read_bytes())
            }
            form = {
                "model": request.model,
                "max_stickers": request.max_stickers,
                "device": request.device,
            }
            response = self._client.post("/segment", files=files, data=form)
            _raise_for_domain_error(response)
            for item in response.json():
                summary = AssetSummary.model_validate(item)
                content = self._fetch_asset_bytes(summary.id)
                results.append(
                    self._store_asset(
                        summary.id,
                        kind=AssetKind(summary.kind),
                        content=content,
                        ext="png",
                        metadata=summary.metadata,
                        parent_ids=summary.parent_ids,
                        created_by=summary.created_by,
                    )
                )
        return results

    def _fetch_asset_bytes(self, asset_id: str) -> bytes:
        response = self._client.get(f"/assets/{asset_id}")
        _raise_for_domain_error(response)
        return response.content

    def depth(self, request: DepthRequest) -> list[Asset]:
        results = []
        for input_asset in request.inputs:
            files = {
                "image": (input_asset.content_path.name, input_asset.content_path.read_bytes())
            }
            form = {"model": request.model, "device": request.device}
            response = self._client.post("/depth", files=files, data=form)
            _raise_for_domain_error(response)
            results.append(
                self._store_asset(
                    response.headers["X-Splat-Asset-Id"],
                    kind=AssetKind.DEPTH_MAP,
                    content=response.content,
                    ext="npy",
                    metadata={},
                    parent_ids=[input_asset.id],
                    created_by=f"depth:{request.model}",
                )
            )
        return results

    def mesh(self, request: MeshRequest) -> list[Asset]:
        results = []
        for input_asset in request.inputs:
            files = {
                "image": (input_asset.content_path.name, input_asset.content_path.read_bytes())
            }
            form = {"model": request.model, "device": request.device}
            response = self._client.post("/mesh", files=files, data=form)
            _raise_for_domain_error(response)
            results.append(
                self._store_asset(
                    response.headers["X-Splat-Asset-Id"],
                    kind=AssetKind.SHAPE_3D,
                    content=response.content,
                    ext="glb",
                    metadata={},
                    parent_ids=[input_asset.id],
                    created_by=f"mesh:{request.model}",
                )
            )
        return results

    def gaussian(self, request: GaussianRequest) -> GaussianResult:
        files = [("images", (p.name, p.read_bytes())) for p in request.inputs]
        form = {
            "model": request.model,
            "device": request.device,
            "to": request.output_path.suffix.lstrip("."),
        }
        response = self._client.post("/gaussian", files=files, data=form)
        _raise_for_domain_error(response)
        request.output_path.write_bytes(response.content)
        cloud = get_reader(request.output_path.suffix).read(request.output_path)
        warnings = response.headers.get("X-Splat-Warnings", "")
        return GaussianResult(cloud=cloud, warnings=warnings.split("; ") if warnings else [])

    def tools_convert(self, request: ConvertRequest) -> ConvertResult:
        files = {"input": (request.input_path.name, request.input_path.read_bytes())}
        form = {"to": request.output_path.suffix.lstrip(".")}
        if request.from_format:
            form["from_format"] = request.from_format
        response = self._client.post("/convert", files=files, data=form)
        _raise_for_domain_error(response)
        request.output_path.write_bytes(response.content)
        cloud = get_reader(request.output_path.suffix).read(request.output_path)
        warnings = response.headers.get("X-Splat-Warnings", "")
        return ConvertResult(cloud=cloud, warnings=warnings.split("; ") if warnings else [])

    def tools_compress(self, request: CompressRequest) -> GaussianCloud:
        files = {"input": (request.input_path.name, request.input_path.read_bytes())}
        form = {"profile": request.profile, "to": request.output_path.suffix.lstrip(".")}
        response = self._client.post("/compress", files=files, data=form)
        _raise_for_domain_error(response)
        request.output_path.write_bytes(response.content)
        return get_reader(request.output_path.suffix).read(request.output_path)

    def info(self, path: Path) -> InfoSummary:
        files = {"input": (path.name, path.read_bytes())}
        response = self._client.post("/info", files=files)
        _raise_for_domain_error(response)
        data = InfoResponse.model_validate(response.json())
        return InfoSummary(
            format=data.format,
            points=data.points,
            sh_degree=data.sh_degree,
            bbox_min=data.bbox_min,
            bbox_max=data.bbox_max,
        )

    def validate(self, path: Path, *, strict: bool = False) -> ValidationSummary:
        files = {"input": (path.name, path.read_bytes())}
        response = self._client.post("/validate", files=files, data={"strict": strict})
        _raise_for_domain_error(response)
        data = ValidationResponse.model_validate(response.json())
        return ValidationSummary(valid=data.valid, issues=data.issues, points=data.points)

    def models_list(self) -> list[ModelSummary]:
        response = self._client.get("/models")
        _raise_for_domain_error(response)
        rows = [WireModelSummary.model_validate(item) for item in response.json()]
        return [
            ModelSummary(name=r.name, runtime=r.runtime, license=r.license, cached=r.cached)
            for r in rows
        ]

    def models_pull(self, name: str) -> None:
        response = self._client.post(f"/models/{name}/pull")
        _raise_for_domain_error(response)

    def models_info(self, name: str) -> ModelInfo:
        response = self._client.get(f"/models/{name}")
        _raise_for_domain_error(response)
        data = ModelInfoResponse.model_validate(response.json())
        return ModelInfo(
            name=data.name,
            runtime=data.runtime,
            source=data.source,
            license=data.license,
            min_images=data.min_images,
            max_images=data.max_images,
        )

    def models_rm(self, name: str) -> None:
        response = self._client.delete(f"/models/{name}")
        _raise_for_domain_error(response)
