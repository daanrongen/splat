from fastapi import APIRouter, File, Form, Response, UploadFile

from splat.domain.manifest import ManifestKind
from splat.handlers.depth import DepthRequest, handle
from splat.http._files import saved_upload
from splat.registry.wiring import get_manifest_repository

router = APIRouter()


@router.post("/depth")
def depth(
    image: UploadFile = File(...),
    model: str = Form("depth-pro"),
    device: str = Form("auto"),
) -> Response:
    cache = get_manifest_repository()
    with saved_upload(image) as path:
        asset = cache.put_external(path, kind=ManifestKind.IMAGE)
    result = handle(DepthRequest(inputs=[asset], model=model, device=device))[0]
    return Response(
        content=result.content_path.read_bytes(),
        media_type="application/octet-stream",
        headers={"X-Splat-Asset-Id": result.id},
    )
