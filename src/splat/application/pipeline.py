"""Cache-aware orchestration for the generative pipeline stages
(diffuse/segment/depth/...). Each `run_*` function computes a deterministic
cache key from its inputs, short-circuits on a cache hit, and otherwise runs
the plain use case and stores the result — this is what makes
`stage -> stage -> stage` chains memoize instead of recomputing on rerun.
"""

import hashlib
import json
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from splat.application.depth import EstimateDepthUseCase
from splat.application.diffuse import DiffuseUseCase
from splat.application.mesh import PredictMeshUseCase
from splat.application.segment import SegmentUseCase
from splat.application.tools import displace_height
from splat.domain.asset import Asset, AssetKind
from splat.domain.errors import SplatDomainError
from splat.domain.image_space import DepthMap, Shape3D
from splat.ports.asset_cache import AssetCache
from splat.ports.depth import DepthEstimationBackend
from splat.ports.diffusion import DiffusionBackend
from splat.ports.mesh import MeshPredictionBackend
from splat.ports.segmentation import SegmentationBackend


def compute_cache_key(
    *, stage: str, model: str, params: dict, parent_ids: tuple[str, ...] = ()
) -> str:
    payload = json.dumps(
        {"stage": stage, "model": model, "params": params, "parents": sorted(parent_ids)},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def run_diffuse(
    backend: DiffusionBackend,
    cache: AssetCache,
    *,
    model_name: str,
    prompt: str,
    params: dict,
) -> Asset:
    cache_key = compute_cache_key(
        stage="diffuse", model=model_name, params={"prompt": prompt, **params}
    )
    if (hit := cache.find(cache_key)) is not None:
        return hit

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "diffused.png"
        DiffuseUseCase(backend).execute(prompt, output_path=tmp_path, **params)
        content_bytes = tmp_path.read_bytes()

    return cache.put(
        cache_key,
        kind=AssetKind.IMAGE,
        content_bytes=content_bytes,
        ext="png",
        metadata={"prompt": prompt, **params},
        parent_ids=[],
        created_by=f"diffuse:{model_name}",
    )


def run_segment(
    backend: SegmentationBackend,
    cache: AssetCache,
    *,
    model_name: str,
    input_asset: Asset,
    params: dict,
) -> list[Asset]:
    cache_key = compute_cache_key(
        stage="segment", model=model_name, params=params, parent_ids=(input_asset.id,)
    )
    manifest = cache.find(cache_key)
    if manifest is not None:
        return [cache.get(child_id) for child_id in manifest.metadata["children"]]

    stickers = SegmentUseCase(backend).execute(input_asset.content_path, **params)

    children: list[Asset] = []
    for i, sticker in enumerate(stickers):
        buf = BytesIO()
        Image.fromarray(sticker.rgba, mode="RGBA").save(buf, format="PNG")
        child_id = f"{cache_key}-{i:03d}"
        children.append(
            cache.put(
                child_id,
                kind=AssetKind.STICKER,
                content_bytes=buf.getvalue(),
                ext="png",
                metadata={"bbox": sticker.bbox, "score": sticker.score, "area": sticker.area},
                parent_ids=[input_asset.id],
                created_by=f"segment:{model_name}",
            )
        )

    # A zero-byte marker asset recording the fan-out, so a rerun short-circuits.
    cache.put(
        cache_key,
        kind=AssetKind.STICKER,
        content_bytes=b"",
        ext="manifest",
        metadata={"children": [c.id for c in children]},
        parent_ids=[input_asset.id],
        created_by=f"segment:{model_name}",
    )
    return children


def run_depth(
    backend: DepthEstimationBackend,
    cache: AssetCache,
    *,
    model_name: str,
    input_asset: Asset,
    params: dict,
) -> Asset:
    cache_key = compute_cache_key(
        stage="depth", model=model_name, params=params, parent_ids=(input_asset.id,)
    )
    if (hit := cache.find(cache_key)) is not None:
        return hit

    depth_map = EstimateDepthUseCase(backend).execute(input_asset.content_path, **params)

    buf = BytesIO()
    np.save(buf, depth_map.depth)

    return cache.put(
        cache_key,
        kind=AssetKind.DEPTH_MAP,
        content_bytes=buf.getvalue(),
        ext="npy",
        metadata={
            "focal_length_px": depth_map.focal_length_px,
            "field_of_view_deg": depth_map.field_of_view_deg,
            **depth_map.metadata,
        },
        parent_ids=[input_asset.id],
        created_by=f"depth:{model_name}",
    )


def _shape_to_mesh_bytes(shape: Shape3D, export_format: str) -> bytes:
    mesh = trimesh.Trimesh(vertices=shape.vertices, faces=shape.faces, process=False)
    if shape.uv is not None and shape.texture is not None:
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=shape.uv, image=Image.fromarray(shape.texture)
        )
    buf = BytesIO()
    try:
        mesh.export(buf, file_type=export_format)
    except Exception as exc:
        raise SplatDomainError(f"Could not export mesh as {export_format!r}: {exc}") from exc
    return buf.getvalue()


def run_mesh(
    backend: MeshPredictionBackend,
    cache: AssetCache,
    *,
    model_name: str,
    input_asset: Asset,
    params: dict,
) -> Asset:
    cache_key = compute_cache_key(
        stage="mesh", model=model_name, params=params, parent_ids=(input_asset.id,)
    )
    if (hit := cache.find(cache_key)) is not None:
        return hit

    shape = PredictMeshUseCase(backend).execute(input_asset.content_path, **params)

    return cache.put(
        cache_key,
        kind=AssetKind.SHAPE_3D,
        content_bytes=_shape_to_mesh_bytes(shape, "glb"),
        ext="glb",
        metadata={**shape.metadata},
        parent_ids=[input_asset.id],
        created_by=f"mesh:{model_name}",
    )


def run_displace_height(
    cache: AssetCache,
    *,
    image_asset: Asset,
    depth_asset: Asset,
    depth_map: DepthMap,
    params: dict,
    export_format: str = "glb",
) -> Asset:
    parent_ids = (image_asset.id, depth_asset.id)
    cache_key = compute_cache_key(
        stage="tools",
        model="displace.height",
        params={"format": export_format, **params},
        parent_ids=parent_ids,
    )
    if (hit := cache.find(cache_key)) is not None:
        return hit

    shape = displace_height.execute(image_asset.content_path, depth_map, **params)

    return cache.put(
        cache_key,
        kind=AssetKind.SHAPE_3D,
        content_bytes=_shape_to_mesh_bytes(shape, export_format),
        ext=export_format,
        metadata={**shape.metadata},
        parent_ids=list(parent_ids),
        created_by="tools:displace.height",
    )
