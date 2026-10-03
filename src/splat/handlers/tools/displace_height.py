from dataclasses import dataclass

import numpy as np

from splat.application.pipeline import run_displace_height
from splat.domain.errors import SplatDomainError
from splat.domain.image_space import DepthMap
from splat.domain.manifest import Manifest, ManifestKind
from splat.ports.manifest_repository import ManifestRepository
from splat.registry.wiring import get_manifest_repository


@dataclass(frozen=True)
class DisplaceHeightRequest:
    inputs: list[Manifest]
    export_format: str = "glb"


def handle(request: DisplaceHeightRequest) -> list[Manifest]:
    cache = get_manifest_repository()
    return [_displace_height_one(asset, cache, request.export_format) for asset in request.inputs]


def _displace_height_one(
    asset: Manifest, cache: ManifestRepository, export_format: str
) -> Manifest:
    if asset.kind != ManifestKind.DEPTH_MAP:
        raise SplatDomainError(
            "displace.height requires a depth map - pipe through `splat depth` first, "
            "e.g. `splat depth image.png | splat tools displace.height -o out.glb`."
        )
    if asset.metadata.units != "metres":
        raise SplatDomainError(
            f"displace.height requires metric depth in metres, but {asset.id} is relative "
            f"{asset.metadata.units} ({asset.created_by}); use a metric model such as depth-pro."
        )
    if not asset.parent_ids:
        raise SplatDomainError(
            f"Depth asset {asset.id} has no source image to texture the mesh with."
        )
    image_asset = cache.get(asset.parent_ids[0])
    depth_map = DepthMap(
        depth=np.load(asset.content_path),
        focal_length_px=asset.metadata.focal_length_px,
        field_of_view_deg=asset.metadata.field_of_view_deg,
        metadata=asset.metadata.extra,
    )
    return run_displace_height(
        cache,
        image_asset=image_asset,
        depth_asset=asset,
        depth_map=depth_map,
        params={},
        export_format=export_format,
    )
