from fastapi import APIRouter, Response

from splat.registry.wiring import get_manifest_repository

router = APIRouter()


@router.get("/assets/{asset_id}")
def get_asset(asset_id: str) -> Response:
    cache = get_manifest_repository()
    asset = cache.get(asset_id)
    return Response(content=asset.content_path.read_bytes(), media_type="application/octet-stream")
