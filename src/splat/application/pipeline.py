"""Cache-aware orchestration for the generative pipeline stages
(diffuse/segment/depth/...). Each `run_*` function computes a deterministic
cache key from its inputs, short-circuits on a cache hit, and otherwise runs
the plain use case and stores the result — this is what makes
`stage -> stage -> stage` chains memoize instead of recomputing on rerun.
"""

import hashlib
import json
import tempfile
from collections.abc import Callable
from io import BytesIO
from pathlib import Path

import numpy as np
import trimesh

from splat.adapters.formats.image import decode_rgb_or_rgba, encode_png, read_rgb_or_rgba
from splat.adapters.formats.ply import PlyWriter
from splat.application.caption import CaptionUseCase
from splat.application.depth import EstimateDepthUseCase
from splat.application.diffuse import DiffuseUseCase
from splat.application.embed import EmbedUseCase
from splat.application.reconstruct import ReconstructUseCase
from splat.application.render import RenderUseCase
from splat.application.segment import SegmentUseCase
from splat.application.upscale import UpscaleUseCase
from splat.domain.errors import SplatDomainError
from splat.domain.gaussians import (
    GaussianCloud,
    normalize_gaussian_cloud,
    to_convention,
    with_view,
)
from splat.domain.image_space import Shape3D
from splat.domain.manifest import Manifest, ManifestKind
from splat.domain.manifest_metadata import (
    CaptionMetadata,
    DepthMetadata,
    EmbeddingMetadata,
    MeshMetadata,
    RasterMetadata,
    SegmentManifestMetadata,
    StickerMetadata,
)
from splat.ports.caption import CaptioningBackend
from splat.ports.depth import DepthEstimationBackend
from splat.ports.diffusion import DiffusionBackend
from splat.ports.embedding import EmbeddingBackend
from splat.ports.manifest_repository import ManifestRepository
from splat.ports.reconstruction import ReconstructionBackend
from splat.ports.render import RenderBackend
from splat.ports.segmentation import SegmentationBackend
from splat.ports.upscaling import UpscalingBackend
from splat.registry.wiring import get_reader


def compute_cache_key(
    *, stage: str, model: str, params: dict, parent_ids: tuple[str, ...] = ()
) -> str:
    # Where a stage runs doesn't change what it produces.
    params = {k: v for k, v in params.items() if k != "device"}
    payload = json.dumps(
        {"stage": stage, "model": model, "params": params, "parents": sorted(parent_ids)},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def run_diffuse(
    backend: DiffusionBackend,
    cache: ManifestRepository,
    *,
    model_name: str,
    prompt: str,
    params: dict,
    input_asset: Manifest | None = None,
) -> Manifest:
    invocation = {"prompt": prompt, **params}
    parent_ids = (input_asset.id,) if input_asset is not None else ()
    cache_key = compute_cache_key(
        stage="diffuse", model=model_name, params=invocation, parent_ids=parent_ids
    )
    if (hit := cache.find(cache_key)) is not None:
        return hit

    source_image = read_rgb_or_rgba(input_asset.content_path) if input_asset is not None else None

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "diffused.png"
        DiffuseUseCase(backend).execute(prompt, output_path=tmp_path, image=source_image, **params)
        content_bytes = tmp_path.read_bytes()

    output = decode_rgb_or_rgba(content_bytes)
    return cache.put(
        cache_key,
        kind=ManifestKind.IMAGE,
        content_bytes=content_bytes,
        ext="png",
        metadata=RasterMetadata(
            source_width=source_image.shape[1] if source_image is not None else None,
            source_height=source_image.shape[0] if source_image is not None else None,
            output_width=output.shape[1],
            output_height=output.shape[0],
        ),
        params=invocation,
        parent_ids=list(parent_ids),
        created_by=f"diffuse:{model_name}",
    )


def run_segment(
    backend: SegmentationBackend,
    cache: ManifestRepository,
    *,
    model_name: str,
    input_asset: Manifest,
    params: dict,
) -> list[Manifest]:
    cache_key = compute_cache_key(
        stage="segment", model=model_name, params=params, parent_ids=(input_asset.id,)
    )
    manifest = cache.find(cache_key)
    if manifest is not None:
        return [cache.get(child_id) for child_id in manifest.metadata.children]

    stickers = SegmentUseCase(backend).execute(input_asset.content_path, **params)

    children: list[Manifest] = []
    for i, sticker in enumerate(stickers):
        child_id = f"{cache_key}-{i:03d}"
        children.append(
            cache.put(
                child_id,
                kind=ManifestKind.STICKER,
                content_bytes=encode_png(sticker.rgba),
                ext="png",
                metadata=StickerMetadata(
                    bbox=sticker.bbox,
                    score=sticker.score,
                    area=sticker.area,
                    width=sticker.rgba.shape[1],
                    height=sticker.rgba.shape[0],
                ),
                params=params,
                parent_ids=[input_asset.id],
                created_by=f"segment:{model_name}",
            )
        )

    # A zero-byte marker asset recording the fan-out, so a rerun short-circuits.
    cache.put(
        cache_key,
        kind=ManifestKind.FAN_OUT,
        content_bytes=b"",
        ext="manifest",
        metadata=SegmentManifestMetadata(children=[c.id for c in children]),
        params=params,
        parent_ids=[input_asset.id],
        created_by=f"segment:{model_name}",
    )
    return children


def run_depth(
    backend: DepthEstimationBackend,
    cache: ManifestRepository,
    *,
    model_name: str,
    input_asset: Manifest,
    params: dict,
) -> Manifest:
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
        kind=ManifestKind.DEPTH_MAP,
        content_bytes=buf.getvalue(),
        ext="npy",
        metadata=DepthMetadata(
            focal_length_px=depth_map.focal_length_px,
            field_of_view_deg=depth_map.field_of_view_deg,
            width=depth_map.depth.shape[1],
            height=depth_map.depth.shape[0],
            units=depth_map.units,
            extra=depth_map.metadata,
        ),
        params=params,
        parent_ids=[input_asset.id],
        created_by=f"depth:{model_name}",
    )


def run_caption(
    backend: CaptioningBackend,
    cache: ManifestRepository,
    *,
    model_name: str,
    input_asset: Manifest,
    params: dict,
) -> Manifest:
    cache_key = compute_cache_key(
        stage="caption", model=model_name, params=params, parent_ids=(input_asset.id,)
    )
    if (hit := cache.find(cache_key)) is not None:
        return hit

    text = CaptionUseCase(backend).execute(input_asset.content_path, **params).strip()
    content_bytes = text.encode("utf-8")

    return cache.put(
        cache_key,
        kind=ManifestKind.CAPTION,
        content_bytes=content_bytes,
        ext="txt",
        metadata=CaptionMetadata(text_length=len(text), model=model_name),
        params=params,
        parent_ids=[input_asset.id],
        created_by=f"caption:{model_name}",
    )


def _embedding_metadata(
    embedding: np.ndarray,
    *,
    input_type: str,
    model_name: str,
    text_sha256: str | None = None,
    text_length: int | None = None,
) -> EmbeddingMetadata:
    return EmbeddingMetadata(
        input_type=input_type,
        dtype=str(embedding.dtype),
        shape=list(embedding.shape),
        dimension=int(embedding.shape[0]),
        normalized=True,
        model=model_name,
        text_sha256=text_sha256,
        text_length=text_length,
    )


def _embedding_bytes(embedding: np.ndarray) -> bytes:
    buf = BytesIO()
    np.save(buf, embedding.astype(np.float32, copy=False))
    return buf.getvalue()


def run_embed_image(
    backend: EmbeddingBackend,
    cache: ManifestRepository,
    *,
    model_name: str,
    input_asset: Manifest,
    params: dict,
) -> Manifest:
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
        kind=ManifestKind.EMBEDDING,
        content_bytes=_embedding_bytes(embedding),
        ext="npy",
        metadata=_embedding_metadata(embedding, input_type="image", model_name=model_name),
        params=params,
        parent_ids=[input_asset.id],
        created_by=f"embed:{model_name}",
    )


def run_embed_text(
    backend: EmbeddingBackend,
    cache: ManifestRepository,
    *,
    model_name: str,
    text: str,
    params: dict,
    input_asset: Manifest | None = None,
) -> Manifest:
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
        kind=ManifestKind.EMBEDDING,
        content_bytes=_embedding_bytes(embedding),
        ext="npy",
        metadata=_embedding_metadata(
            embedding,
            input_type="text",
            model_name=model_name,
            text_sha256=text_sha256,
            text_length=len(text),
        ),
        params=params,
        parent_ids=list(parent_ids),
        created_by=f"embed:{model_name}",
    )


def run_upscale(
    backend: UpscalingBackend,
    cache: ManifestRepository,
    *,
    model_name: str,
    input_asset: Manifest,
    params: dict,
) -> Manifest:
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
        kind=ManifestKind.IMAGE,
        content_bytes=encode_png(result.image),
        ext="png",
        metadata=RasterMetadata(
            variant=variant,
            source_width=result.source_width,
            source_height=result.source_height,
            output_width=result.output_width,
            output_height=result.output_height,
        ),
        params={"model": model_name, **params},
        parent_ids=[input_asset.id],
        created_by=f"upscale:{model_name}",
    )


def shape_to_mesh_bytes(shape: Shape3D, export_format: str) -> bytes:
    mesh = trimesh.Trimesh(vertices=shape.vertices, faces=shape.faces, process=False)
    if shape.uv is not None and shape.texture is not None:
        mesh.visual = trimesh.visual.TextureVisuals(uv=shape.uv, image=shape.texture)
    buf = BytesIO()
    try:
        mesh.export(buf, file_type=export_format)
    except Exception as exc:
        raise SplatDomainError(f"Could not export mesh as {export_format!r}: {exc}") from exc
    return buf.getvalue()


def _gaussian_to_ply_bytes(cloud: GaussianCloud) -> bytes:
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "cloud.ply"
        PlyWriter().write(cloud, path)
        return path.read_bytes()


def run_gaussian(
    backend: ReconstructionBackend,
    cache: ManifestRepository,
    *,
    model_name: str,
    input_assets: list[Manifest],
    params: dict,
    on_progress: Callable[[str], None] | None = None,
) -> Manifest:
    parent_ids = tuple(asset.id for asset in input_assets)
    cache_key = compute_cache_key(
        stage="gaussian", model=model_name, params=params, parent_ids=parent_ids
    )
    if (hit := cache.find(cache_key)) is not None:
        return hit

    execute_params = {k: v for k, v in params.items() if k not in ("device", "declutter")}
    cloud = ReconstructUseCase(backend).execute(
        [asset.content_path for asset in input_assets],
        device=params.get("device", "auto"),
        on_progress=on_progress,
        **execute_params,
    )
    if params.get("declutter"):
        from splat.adapters.cleanup.density_declutter import DensityDeclutterer

        cloud = DensityDeclutterer().declutter(cloud)
    # SfM and most feed-forward reconstruction have no absolute scale, so the
    # default is to normalize; a backend predicting metres says so and keeps it.
    if not backend.provides_metric_scale:
        cloud = normalize_gaussian_cloud(cloud)
    cloud.metadata.metric_scale = backend.provides_metric_scale
    cloud.metadata.source_model = model_name
    cloud.metadata.source_format = "ply"
    cloud.metadata.license = backend.license
    # Reconstruction backends calibrate/triangulate in the OpenCV/COLMAP frame
    # (+X right, +Y down, +Z forward), so that is what they hand back. Storing
    # it that way was leaving every cloud upside down with the camera facing
    # away from the scene in any consumer that assumes Y-up: COLMAP's up is
    # -Y and its scene sits at +Z, while a default OpenGL-style camera looks
    # down -Z. `up_axis` could not describe it either, its type only admits
    # "y" or "z". Converting on the way in gives one canonical stored
    # convention whose metadata is true, and `render` skips its own flip.
    cloud = to_convention(cloud, "opengl")
    content_bytes = _gaussian_to_ply_bytes(cloud)

    return cache.put(
        cache_key,
        kind=ManifestKind.GAUSSIAN_CLOUD,
        content_bytes=content_bytes,
        ext="ply",
        metadata=cloud.metadata,
        params={"model": model_name, **params},
        parent_ids=list(parent_ids),
        created_by=f"gaussian:{model_name}",
    )


def run_render(
    backend: RenderBackend,
    cache: ManifestRepository,
    *,
    model_name: str,
    input_asset: Manifest,
    params: dict,
    on_progress: Callable[[str], None] | None = None,
) -> Manifest:
    cache_key = compute_cache_key(
        stage="render", model=model_name, params=params, parent_ids=(input_asset.id,)
    )
    if (hit := cache.find(cache_key)) is not None:
        return hit

    # Via the registry, not PlyReader: RENDER_CONTRACT accepts any splat_3d
    # manifest, so `splat render scene.spz` has to work as well as `splat info`
    # already does on the same file.
    cloud = get_reader(input_asset.content_path.suffix).read(input_asset.content_path)
    render_params = {k: v for k, v in params.items() if k != "view"}
    if params.get("view") is not None:
        cloud = with_view(cloud, params["view"])
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "render.png"
        RenderUseCase(backend).execute(cloud, tmp_path, on_progress=on_progress, **render_params)
        content_bytes = tmp_path.read_bytes()

    output = decode_rgb_or_rgba(content_bytes)
    return cache.put(
        cache_key,
        kind=ManifestKind.IMAGE,
        content_bytes=content_bytes,
        ext="png",
        metadata=RasterMetadata(output_width=output.shape[1], output_height=output.shape[0]),
        params=params,
        parent_ids=[input_asset.id],
        created_by=f"render:{model_name}",
    )


def run_mesh(
    cache: ManifestRepository,
    *,
    model_name: str,
    parent_ids: tuple[str, ...],
    params: dict,
    build: Callable[[], tuple[bytes, dict]],
) -> Manifest:
    """Caches a `shape_3d` from whichever mesher `build` runs; it returns the
    encoded mesh in `params["format"]` plus its metadata."""
    cache_key = compute_cache_key(
        stage="mesh", model=model_name, params=params, parent_ids=parent_ids
    )
    if (hit := cache.find(cache_key)) is not None:
        return hit

    content, metadata = build()
    return cache.put(
        cache_key,
        kind=ManifestKind.SHAPE_3D,
        content_bytes=content,
        ext=params["format"],
        metadata=MeshMetadata(extra=metadata),
        params=params,
        parent_ids=list(parent_ids),
        created_by=f"mesh:{model_name}",
    )
