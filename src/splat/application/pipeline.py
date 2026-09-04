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

from splat.adapters.formats.ply import PlyWriter
from splat.application.caption import CaptionUseCase
from splat.application.depth import EstimateDepthUseCase
from splat.application.diffuse import DiffuseUseCase
from splat.application.embed import EmbedUseCase
from splat.application.mesh import PredictMeshUseCase
from splat.application.reconstruct import ReconstructUseCase
from splat.application.segment import SegmentUseCase
from splat.application.tools import displace_height
from splat.application.upscale import UpscaleUseCase
from splat.domain.asset import Asset, AssetKind
from splat.domain.errors import SplatDomainError
from splat.domain.gaussians import GaussianCloud
from splat.domain.image_space import DepthMap, Shape3D
from splat.image_io import encode_png
from splat.ports.asset_cache import AssetCache
from splat.ports.caption import CaptioningBackend
from splat.ports.depth import DepthEstimationBackend
from splat.ports.diffusion import DiffusionBackend
from splat.ports.embedding import EmbeddingBackend
from splat.ports.mesh import MeshPredictionBackend
from splat.ports.reconstruction import ReconstructionBackend
from splat.ports.segmentation import SegmentationBackend
from splat.ports.upscaling import UpscalingBackend


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
        child_id = f"{cache_key}-{i:03d}"
        children.append(
            cache.put(
                child_id,
                kind=AssetKind.STICKER,
                content_bytes=encode_png(sticker.rgba),
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


def run_caption(
    backend: CaptioningBackend,
    cache: AssetCache,
    *,
    model_name: str,
    input_asset: Asset,
    params: dict,
) -> Asset:
    cache_key = compute_cache_key(
        stage="caption", model=model_name, params=params, parent_ids=(input_asset.id,)
    )
    if (hit := cache.find(cache_key)) is not None:
        return hit

    text = CaptionUseCase(backend).execute(input_asset.content_path, **params).strip()
    content_bytes = text.encode("utf-8")

    return cache.put(
        cache_key,
        kind=AssetKind.CAPTION,
        content_bytes=content_bytes,
        ext="txt",
        metadata={**params, "text_length": len(text)},
        parent_ids=[input_asset.id],
        created_by=f"caption:{model_name}",
    )


def _embedding_metadata(
    embedding: np.ndarray,
    *,
    input_type: str,
    model_name: str,
    params: dict,
    extra: dict,
) -> dict:
    return {
        **params,
        **extra,
        "input_type": input_type,
        "dtype": str(embedding.dtype),
        "shape": list(embedding.shape),
        "dimension": int(embedding.shape[0]),
        "normalized": True,
        "model": model_name,
    }


def _embedding_bytes(embedding: np.ndarray) -> bytes:
    buf = BytesIO()
    np.save(buf, embedding.astype(np.float32, copy=False))
    return buf.getvalue()


def run_embed_image(
    backend: EmbeddingBackend,
    cache: AssetCache,
    *,
    model_name: str,
    input_asset: Asset,
    params: dict,
) -> Asset:
    cache_key = compute_cache_key(
        stage="embed",
        model=model_name,
        params={"input_type": "image", **params},
        parent_ids=(input_asset.id,),
    )
    if (hit := cache.find(cache_key)) is not None:
        return hit

    embedding = EmbedUseCase(backend).image(input_asset.content_path, **params)

    return cache.put(
        cache_key,
        kind=AssetKind.EMBEDDING,
        content_bytes=_embedding_bytes(embedding),
        ext="npy",
        metadata=_embedding_metadata(
            embedding,
            input_type="image",
            model_name=model_name,
            params=params,
            extra={},
        ),
        parent_ids=[input_asset.id],
        created_by=f"embed:{model_name}",
    )


def run_embed_text(
    backend: EmbeddingBackend,
    cache: AssetCache,
    *,
    model_name: str,
    text: str,
    params: dict,
    input_asset: Asset | None = None,
) -> Asset:
    text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    parent_ids = (input_asset.id,) if input_asset is not None else ()
    key_params = {"input_type": "text", **params}
    if input_asset is None:
        key_params["text_sha256"] = text_sha256
    cache_key = compute_cache_key(
        stage="embed",
        model=model_name,
        params=key_params,
        parent_ids=parent_ids,
    )
    if (hit := cache.find(cache_key)) is not None:
        return hit

    embedding = EmbedUseCase(backend).text(text, **params)

    return cache.put(
        cache_key,
        kind=AssetKind.EMBEDDING,
        content_bytes=_embedding_bytes(embedding),
        ext="npy",
        metadata=_embedding_metadata(
            embedding,
            input_type="text",
            model_name=model_name,
            params=params,
            extra={"text_sha256": text_sha256, "text_length": len(text)},
        ),
        parent_ids=list(parent_ids),
        created_by=f"embed:{model_name}",
    )


def run_upscale(
    backend: UpscalingBackend,
    cache: AssetCache,
    *,
    model_name: str,
    input_asset: Asset,
    params: dict,
) -> Asset:
    cache_key = compute_cache_key(
        stage="upscale", model=model_name, params=params, parent_ids=(input_asset.id,)
    )
    if (hit := cache.find(cache_key)) is not None:
        return hit

    result = UpscaleUseCase(backend).execute(input_asset.content_path, **params)
    variant = None
    if hasattr(backend, "variant_for_factor"):
        variant = backend.variant_for_factor(params["factor"])

    return cache.put(
        cache_key,
        kind=AssetKind.IMAGE,
        content_bytes=encode_png(result.image),
        ext="png",
        metadata={
            "model": model_name,
            "variant": variant,
            "factor": params["factor"],
            "tile": params.get("tile", 0),
            "source_width": result.source_width,
            "source_height": result.source_height,
            "output_width": result.output_width,
            "output_height": result.output_height,
        },
        parent_ids=[input_asset.id],
        created_by=f"upscale:{model_name}",
    )


def _shape_to_mesh_bytes(shape: Shape3D, export_format: str) -> bytes:
    mesh = trimesh.Trimesh(vertices=shape.vertices, faces=shape.faces, process=False)
    if shape.uv is not None and shape.texture is not None:
        mesh.visual = trimesh.visual.TextureVisuals(uv=shape.uv, image=shape.texture)
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


def _gaussian_to_ply_bytes(cloud: GaussianCloud) -> bytes:
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "cloud.ply"
        PlyWriter().write(cloud, path)
        return path.read_bytes()


def run_gaussian(
    backend: ReconstructionBackend,
    cache: AssetCache,
    *,
    model_name: str,
    input_assets: list[Asset],
    params: dict,
) -> Asset:
    parent_ids = tuple(asset.id for asset in input_assets)
    cache_key = compute_cache_key(
        stage="gaussian", model=model_name, params=params, parent_ids=parent_ids
    )
    if (hit := cache.find(cache_key)) is not None:
        return hit

    execute_params = {key: value for key, value in params.items() if key != "device"}
    cloud = ReconstructUseCase(backend).execute(
        [asset.content_path for asset in input_assets],
        device=params.get("device", "auto"),
        **execute_params,
    )
    content_bytes = _gaussian_to_ply_bytes(cloud)

    return cache.put(
        cache_key,
        kind=AssetKind.GAUSSIAN_CLOUD,
        content_bytes=content_bytes,
        ext="ply",
        metadata={
            **params,
            "model": model_name,
            "source_model": model_name,
            "source_format": "ply",
            "point_count": cloud.point_count,
            "sh_degree": cloud.sh_degree,
            "scale_activation": cloud.scale_activation,
            "opacity_activation": cloud.opacity_activation,
        },
        parent_ids=list(parent_ids),
        created_by=f"gaussian:{model_name}",
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
