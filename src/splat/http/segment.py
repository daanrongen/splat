from fastapi import APIRouter, File, Form, UploadFile

from splat.domain.manifest import ManifestKind
from splat.handlers.segment import SegmentRequest, handle
from splat.http._files import saved_upload
from splat.http._schemas import AssetSummary
from splat.registry.wiring import get_asset_cache

router = APIRouter()


@router.post("/segment")
def segment(
    image: UploadFile = File(...),
    model: str = Form("sam-mlx"),
    max_stickers: int = Form(20),
    device: str = Form("auto"),
) -> list[AssetSummary]:
    cache = get_asset_cache()
    with saved_upload(image) as path:
        asset = cache.put_external(path, kind=ManifestKind.IMAGE)
    stickers = handle(
        SegmentRequest(inputs=[asset], model=model, max_stickers=max_stickers, device=device)
    )
    return [AssetSummary.from_asset(a) for a in stickers]
