"""HTTP client for a remote `splat http` server — used by the CLI when
SPLAT_URL is set. Reuses http/_schemas.py's pydantic models as the wire
contract, so there is exactly one definition of each request/response
shape, shared with the server that produces/consumes it.
"""

import hashlib
import json
import os
import tempfile
import warnings
from dataclasses import asdict
from io import BytesIO
from pathlib import Path

import httpx
import numpy as np

from splat import __version__
from splat.adapters.formats.image import decode_rgb_or_rgba, read_rgb_or_rgba
from splat.domain.errors import SplatDomainError
from splat.domain.manifest import Manifest, ManifestKind
from splat.domain.manifest_metadata import (
    CaptionMetadata,
    DepthMetadata,
    EmbeddingMetadata,
    ManifestMetadata,
    RasterMetadata,
    StickerMetadata,
)
from splat.handlers.caption import CaptionRequest
from splat.handlers.depth import DepthRequest
from splat.handlers.diffuse import DiffuseRequest, DiffuseResult
from splat.handlers.embed import EmbedRequest
from splat.handlers.gaussian import GaussianRequest
from splat.handlers.segment import SegmentRequest
from splat.handlers.upscale import UpscaleRequest
from splat.http._schemas import (
    DiffuseBody,
    EmbedBody,
    InfoResponse,
    ManifestDetail,
    ModelInfoResponse,
    UpscaleBody,
    ValidationResponse,
)
from splat.http._schemas import ModelSummary as WireModelSummary
from splat.ports.client import InfoSummary, ModelInfo, ModelSummary, ValidationSummary
from splat.registry.wiring import get_manifest_repository, get_reader


def _raise_for_domain_error(response: httpx.Response) -> None:
    if response.is_success:
        return
    try:
        detail = response.json()["detail"]
    except (ValueError, KeyError, TypeError):
        detail = f"{response.status_code} {response.reason_phrase}"
    raise SplatDomainError(detail if response.status_code == 422 else f"server error: {detail}")


def _check_server_version(client: httpx.Client) -> None:
    try:
        response = client.get("/version")
    except httpx.HTTPError:
        return
    if response.status_code != 200:
        return
    server = response.json()["version"]
    server_major, server_minor = server.split(".")[:2]
    major, minor = __version__.split(".")[:2]
    if server_major != major:
        raise SplatDomainError(f"server runs splat {server}, this client is {__version__}")
    if server_minor != minor:
        warnings.warn(f"server runs splat {server}, this client is {__version__}", stacklevel=2)


class RemoteSplatClient:
    """Proxies every SplatClient method to a remote `splat http` server,
    storing asset-shaped results in the local asset cache under the same
    id the server reports, and writing file-shaped results to the
    caller-supplied local output path — so downstream CLI code (-o
    writing, NDJSON piping) works identically to LocalSplatClient."""

    def __init__(self, base_url: str, *, timeout: float = 300.0) -> None:
        token = os.environ.get("SPLAT_TOKEN", "")
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers={"Authorization": f"Bearer {token}"} if token else None,
        )
        _check_server_version(self._client)

    def _store_asset(
        self,
        asset_id: str,
        *,
        kind: ManifestKind,
        content: bytes,
        ext: str,
        metadata: ManifestMetadata,
        params: dict | None = None,
        parent_ids: list[str],
        created_by: str,
    ) -> Manifest:
        cache = get_manifest_repository()
        existing = cache.find(asset_id)
        if existing is not None:
            return existing
        return cache.put(
            asset_id,
            kind=kind,
            content_bytes=content,
            ext=ext,
            metadata=metadata,
            params=params,
            parent_ids=parent_ids,
            created_by=created_by,
        )

    def diffuse(self, request: DiffuseRequest) -> DiffuseResult:
        body = DiffuseBody(
            prompt=request.prompt,
            model=request.model,
            negative_prompt=request.negative_prompt,
            steps=request.steps,
            strength=request.strength,
            seed=request.seed,
            device=request.device,
        )
        source_image = None
        files = None
        if request.inputs:
            input_asset = request.inputs[0]
            source_image = read_rgb_or_rgba(input_asset.content_path)
            files = {
                "image": (input_asset.content_path.name, input_asset.content_path.read_bytes())
            }
        response = self._client.post("/diffuse", data=body.model_dump(), files=files)
        _raise_for_domain_error(response)
        image = decode_rgb_or_rgba(response.content)
        asset = self._store_asset(
            response.headers["X-Splat-Asset-Id"],
            kind=ManifestKind.IMAGE,
            content=response.content,
            ext="png",
            metadata=RasterMetadata(
                source_width=source_image.shape[1] if source_image is not None else None,
                source_height=source_image.shape[0] if source_image is not None else None,
                output_width=image.shape[1],
                output_height=image.shape[0],
            ),
            params={"prompt": request.prompt},
            parent_ids=[request.inputs[0].id] if request.inputs else [],
            created_by=f"diffuse:{request.model}",
        )
        return DiffuseResult(
            asset=asset, license_warning=response.headers.get("X-Splat-License-Warning")
        )

    def segment(self, request: SegmentRequest) -> list[Manifest]:
        results: list[Manifest] = []
        for input_asset in request.inputs:
            files = {
                "image": (input_asset.content_path.name, input_asset.content_path.read_bytes())
            }
            form = {
                "model": request.model,
                "max_stickers": request.max_stickers,
                "device": request.device,
                "foreground": request.foreground,
                "drop_background": request.drop_background,
                "points": list(request.points),
            }
            if request.box:
                form["box"] = request.box
            response = self._client.post("/segment", files=files, data=form)
            _raise_for_domain_error(response)
            for item in response.json():
                detail = ManifestDetail.model_validate(item)
                content = self._fetch_asset_bytes(detail.id)
                results.append(
                    self._store_asset(
                        detail.id,
                        kind=ManifestKind(detail.kind),
                        content=content,
                        ext="png",
                        metadata=StickerMetadata(**detail.metadata),
                        parent_ids=detail.parent_ids,
                        created_by=detail.created_by,
                    )
                )
        return results

    def _fetch_asset_bytes(self, asset_id: str) -> bytes:
        response = self._client.get(f"/assets/{asset_id}")
        _raise_for_domain_error(response)
        return response.content

    def caption(self, request: CaptionRequest) -> list[Manifest]:
        results = []
        for input_asset in request.inputs:
            files = {
                "image": (input_asset.content_path.name, input_asset.content_path.read_bytes())
            }
            form = {
                "model": request.model,
                "prompt": request.prompt,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "device": request.device,
            }
            response = self._client.post("/caption", files=files, data=form)
            _raise_for_domain_error(response)
            text = response.content.decode("utf-8")
            results.append(
                self._store_asset(
                    response.headers["X-Splat-Asset-Id"],
                    kind=ManifestKind.CAPTION,
                    content=response.content,
                    ext="txt",
                    metadata=CaptionMetadata(text_length=len(text), model=request.model),
                    params={
                        "prompt": request.prompt,
                        "max_tokens": request.max_tokens,
                        "temperature": request.temperature,
                    },
                    parent_ids=[input_asset.id],
                    created_by=f"caption:{request.model}",
                )
            )
        return results

    def depth(self, request: DepthRequest) -> list[Manifest]:
        results = []
        for input_asset in request.inputs:
            files = {
                "image": (input_asset.content_path.name, input_asset.content_path.read_bytes())
            }
            form = {"model": request.model, "device": request.device}
            response = self._client.post("/depth", files=files, data=form)
            _raise_for_domain_error(response)
            depth = np.load(BytesIO(response.content))
            results.append(
                self._store_asset(
                    response.headers["X-Splat-Asset-Id"],
                    kind=ManifestKind.DEPTH_MAP,
                    content=response.content,
                    ext="npy",
                    metadata=DepthMetadata(width=depth.shape[1], height=depth.shape[0]),
                    parent_ids=[input_asset.id],
                    created_by=f"depth:{request.model}",
                )
            )
        return results

    def upscale(self, request: UpscaleRequest) -> list[Manifest]:
        results = []
        body = UpscaleBody(model=request.model, factor=request.factor, tile=request.tile)
        for input_asset in request.inputs:
            source_image = read_rgb_or_rgba(input_asset.content_path)
            files = {
                "image": (input_asset.content_path.name, input_asset.content_path.read_bytes())
            }
            response = self._client.post("/upscale", files=files, data=body.model_dump())
            _raise_for_domain_error(response)
            output_image = decode_rgb_or_rgba(response.content)
            results.append(
                self._store_asset(
                    response.headers["X-Splat-Asset-Id"],
                    kind=ManifestKind.IMAGE,
                    content=response.content,
                    ext="png",
                    metadata=RasterMetadata(
                        source_width=source_image.shape[1],
                        source_height=source_image.shape[0],
                        output_width=output_image.shape[1],
                        output_height=output_image.shape[0],
                    ),
                    params={"model": request.model, "factor": request.factor, "tile": request.tile},
                    parent_ids=[input_asset.id],
                    created_by=f"upscale:{request.model}",
                )
            )
        return results

    def embed(self, request: EmbedRequest) -> list[Manifest]:
        results = []
        if request.text is not None:
            body = EmbedBody(text=request.text, model=request.model, device=request.device)
            response = self._client.post("/embed", data=body.model_dump(exclude_none=True))
            _raise_for_domain_error(response)
            vector = np.load(BytesIO(response.content))
            results.append(
                self._store_asset(
                    response.headers["X-Splat-Asset-Id"],
                    kind=ManifestKind.EMBEDDING,
                    content=response.content,
                    ext="npy",
                    metadata=EmbeddingMetadata(
                        input_type="text",
                        dtype=str(vector.dtype),
                        shape=list(vector.shape),
                        dimension=int(vector.shape[0]),
                        normalized=True,
                        model=request.model,
                        text_sha256=hashlib.sha256(request.text.encode("utf-8")).hexdigest(),
                        text_length=len(request.text),
                    ),
                    params={"device": request.device},
                    parent_ids=[],
                    created_by=f"embed:{request.model}",
                )
            )
            return results

        for input_asset in request.inputs or []:
            files = {
                "image": (input_asset.content_path.name, input_asset.content_path.read_bytes())
            }
            body = EmbedBody(model=request.model, device=request.device)
            response = self._client.post(
                "/embed", files=files, data=body.model_dump(exclude_none=True)
            )
            _raise_for_domain_error(response)
            vector = np.load(BytesIO(response.content))
            results.append(
                self._store_asset(
                    response.headers["X-Splat-Asset-Id"],
                    kind=ManifestKind.EMBEDDING,
                    content=response.content,
                    ext="npy",
                    metadata=EmbeddingMetadata(
                        input_type="image",
                        dtype=str(vector.dtype),
                        shape=list(vector.shape),
                        dimension=int(vector.shape[0]),
                        normalized=True,
                        model=request.model,
                    ),
                    params={"device": request.device},
                    parent_ids=[input_asset.id],
                    created_by=f"embed:{request.model}",
                )
            )
        return results

    def gaussian(self, request: GaussianRequest) -> list[Manifest]:
        files = [
            ("images", (asset.content_path.name, asset.content_path.read_bytes()))
            for asset in request.inputs
        ]
        if request.mask is not None:
            files.append(
                ("mask", (request.mask.content_path.name, request.mask.content_path.read_bytes()))
            )
        form = {
            "model": request.model,
            "device": request.device,
            "to": "ply",
            "quality": request.quality,
            "focal_35mm": request.focal_35mm,
            "poses": request.poses,
            "refine_poses": request.refine_poses,
            "low_memory": request.low_memory,
            "seed": request.seed,
            "declutter": request.declutter,
            "normalize_color": request.normalize_color,
            "score": request.score,
        }
        if request.mask is not None:
            form["mask_metadata"] = json.dumps(asdict(request.mask.metadata))
        if request.iters is not None:
            form["iters"] = request.iters
        if request.max_dim is not None:
            form["max_dim"] = request.max_dim
        if request.sh_degree is not None:
            form["sh_degree"] = request.sh_degree
        if request.orbit_frames is not None:
            form["orbit_frames"] = request.orbit_frames
            form["orbit_degrees"] = request.orbit_degrees
        response = self._client.post("/gaussian", files=files, data=form)
        _raise_for_domain_error(response)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "gaussian.ply"
            path.write_bytes(response.content)
            cloud = get_reader(".ply").read(path)
        cloud.metadata.source_model = request.model
        return [
            self._store_asset(
                response.headers["X-Splat-Asset-Id"],
                kind=ManifestKind.GAUSSIAN_CLOUD,
                content=response.content,
                ext="ply",
                metadata=cloud.metadata,
                params={
                    "model": request.model,
                    "device": request.device,
                    "quality": request.quality,
                    "focal_35mm": request.focal_35mm,
                    "poses": request.poses,
                    "refine_poses": request.refine_poses,
                    "low_memory": request.low_memory,
                    "seed": request.seed,
                },
                parent_ids=[asset.id for asset in request.inputs],
                created_by=f"gaussian:{request.model}",
            )
        ]

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
            coordinate_convention=data.coordinate_convention,
            up_axis=data.up_axis,
            source_model=data.source_model,
            license=data.license,
            capture_camera_count=data.capture_camera_count,
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
            ModelSummary(
                name=r.name,
                stage=r.stage,
                runtime=r.runtime,
                license=r.license,
                commercial=r.commercial,
                cached=r.cached,
            )
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
            dimension=data.dimension,
            normalized=data.normalized,
            notes=data.notes,
        )

    def models_rm(self, name: str) -> None:
        response = self._client.delete(f"/models/{name}")
        _raise_for_domain_error(response)
