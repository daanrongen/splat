from dataclasses import dataclass

import numpy as np

from splat.application.pipeline import run_mesh, shape_to_mesh_bytes
from splat.domain.errors import SplatDomainError
from splat.domain.image_space import DepthMap
from splat.domain.manifest import Manifest, ManifestKind
from splat.ports.manifest_repository import ManifestRepository
from splat.registry.wiring import get_manifest_repository

MESH_MODELS = {ManifestKind.DEPTH_MAP: "heightfield", ManifestKind.GAUSSIAN_CLOUD: "poisson"}


@dataclass(frozen=True)
class MeshRequest:
    inputs: list[Manifest]
    model: str | None = None  # inferred from each input's kind
    format: str = "glb"
    depth: int = 9
    opacity_threshold: float = 0.1


def handle(request: MeshRequest) -> list[Manifest]:
    cache = get_manifest_repository()
    return [_mesh_one(asset, request, cache) for asset in request.inputs]


def _mesh_one(asset: Manifest, request: MeshRequest, cache: ManifestRepository) -> Manifest:
    model = request.model or MESH_MODELS.get(asset.kind)
    if model == "heightfield":
        return _heightfield(asset, request.format, cache)
    if model == "poisson":
        return _poisson(asset, request, cache)
    raise SplatDomainError(
        f"mesh needs a depth_map (heightfield) or gaussian_cloud (poisson), got "
        f"{asset.kind.value}" + (f" with model {model!r}" if model else "") + "."
    )


def _heightfield(asset: Manifest, fmt: str, cache: ManifestRepository) -> Manifest:
    from splat.adapters.mesh.heightfield import heightfield_mesh

    if asset.kind != ManifestKind.DEPTH_MAP:
        raise SplatDomainError(
            "heightfield needs a depth map, e.g. `splat depth image.png | splat mesh - -o out.glb`."
        )
    if asset.metadata.units != "metres":
        raise SplatDomainError(
            f"heightfield needs metric depth in metres, but {asset.id} is relative "
            f"{asset.metadata.units} ({asset.created_by}); use a metric model such as depth-pro."
        )
    if not asset.parent_ids:
        raise SplatDomainError(f"Depth asset {asset.id} has no source image to texture with.")
    image = cache.get(asset.parent_ids[0])
    depth_map = DepthMap(
        depth=np.load(asset.content_path),
        focal_length_px=asset.metadata.focal_length_px,
        field_of_view_deg=asset.metadata.field_of_view_deg,
        metadata=asset.metadata.extra,
    )

    def build() -> tuple[bytes, dict]:
        shape = heightfield_mesh(image.content_path, depth_map)
        return shape_to_mesh_bytes(shape, fmt), shape.metadata

    return run_mesh(
        cache,
        model_name="heightfield",
        parent_ids=(image.id, asset.id),
        params={"format": fmt},
        build=build,
    )


def _poisson(asset: Manifest, request: MeshRequest, cache: ManifestRepository) -> Manifest:
    from splat.adapters.mesh.poisson import poisson_mesh

    if asset.kind != ManifestKind.GAUSSIAN_CLOUD:
        raise SplatDomainError(f"poisson needs a gaussian_cloud, got {asset.kind.value}.")
    params = {
        "format": request.format,
        "depth": request.depth,
        "opacity_threshold": request.opacity_threshold,
    }

    def build() -> tuple[bytes, dict]:
        content, vertices, faces = poisson_mesh(
            asset.as_gaussian_cloud(),
            format=request.format,
            depth=request.depth,
            opacity_threshold=request.opacity_threshold,
        )
        return content, {"vertex_count": vertices, "face_count": faces}

    return run_mesh(cache, model_name="poisson", parent_ids=(asset.id,), params=params, build=build)
