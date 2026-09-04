from fastapi import APIRouter, File, Form, Response, UploadFile

from splat.domain.asset import AssetKind
from splat.handlers.upscale import UpscaleRequest, handle
from splat.http._files import saved_upload
from splat.registry.wiring import get_asset_cache

router = APIRouter()


@router.post("/upscale")
def upscale(
    image: UploadFile = File(...),
    model: str = Form("realesrgan-mlx"),
    factor: int = Form(4),
    tile: int = Form(0),
) -> Response:
    cache = get_asset_cache()
    with saved_upload(image) as path:
        asset = cache.put_external(path, kind=AssetKind.IMAGE)
    result = handle(UpscaleRequest(inputs=[asset], model=model, factor=factor, tile=tile))[0]
    return Response(
        content=result.content_path.read_bytes(),
        media_type="image/png",
        headers={"X-Splat-Asset-Id": result.id},
    )
