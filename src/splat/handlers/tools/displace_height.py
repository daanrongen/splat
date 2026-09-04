from dataclasses import dataclass

import numpy as np

from splat.application.pipeline import run_displace_height
from splat.domain.asset import Asset, AssetKind
from splat.domain.errors import SplatDomainError
from splat.domain.image_space import DepthMap
from splat.ports.asset_cache import AssetCache
from splat.registry.wiring import get_asset_cache


@dataclass(frozen=True)
class DisplaceHeightRequest:
    inputs: list[Asset]
    export_format: str = "glb"


def handle(request: DisplaceHeightRequest) -> list[Asset]:
    cache = get_asset_cache()
    return [_displace_height_one(asset, cache, request.export_format) for asset in request.inputs]


def _displace_height_one(asset: Asset, cache: AssetCache, export_format: str) -> Asset:
    if asset.kind != AssetKind.DEPTH_MAP:
        raise SplatDomainError(
            "displace.height requires a depth map - pipe through `splat depth` first, "
            "e.g. `splat depth image.png | splat tools displace.height -o out.glb`."
        )
    if not asset.parent_ids:
        raise SplatDomainError(
            f"Depth asset {asset.id} has no source image to texture the mesh with."
        )
    image_asset = cache.get(asset.parent_ids[0])
    depth_map = DepthMap(
        depth=np.load(asset.content_path),
        focal_length_px=asset.metadata.get("focal_length_px"),
        field_of_view_deg=asset.metadata.get("field_of_view_deg"),
        metadata=asset.metadata,
    )
    return run_displace_height(
        cache,
        image_asset=image_asset,
        depth_asset=asset,
        depth_map=depth_map,
        params={},
        export_format=export_format,
    )
