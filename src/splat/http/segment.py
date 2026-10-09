from fastapi import APIRouter, File, Form, UploadFile

from splat.domain.manifest import ManifestKind
from splat.handlers.segment import SegmentRequest, handle
from splat.http._files import saved_upload
from splat.http._schemas import ManifestDetail
from splat.registry.wiring import get_manifest_repository

router = APIRouter()


@router.post("/segment")
def segment(
    image: UploadFile = File(...),
    model: str = Form("sam-mlx"),
    max_stickers: int = Form(20),
    device: str = Form("auto"),
    foreground: bool = Form(False),
    drop_background: bool = Form(False),
    points: list[str] = Form([]),
    box: str | None = Form(None),
) -> list[ManifestDetail]:
    cache = get_manifest_repository()
    with saved_upload(image) as path:
        asset = cache.put_external(path, kind=ManifestKind.IMAGE)
    stickers = handle(
        SegmentRequest(
            inputs=[asset],
            model=model,
            max_stickers=max_stickers,
            device=device,
            foreground=foreground,
            drop_background=drop_background,
            points=tuple(points),
            box=box,
        )
    )
    return [ManifestDetail.from_manifest(a) for a in stickers]
