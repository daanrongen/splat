from fastapi import APIRouter, File, Form, Response, UploadFile

from splat.domain.manifest import ManifestKind
from splat.handlers.embed import EmbedRequest, handle
from splat.http._files import saved_upload
from splat.registry.wiring import get_asset_cache

router = APIRouter()


@router.post("/embed")
def embed(
    image: UploadFile | None = File(None),
    text: str | None = Form(None),
    model: str = Form("mobileclip2-s0"),
    device: str = Form("auto"),
) -> Response:
    cache = get_asset_cache()
    inputs = None
    if image is not None:
        with saved_upload(image) as path:
            asset = cache.put_external(path, kind=ManifestKind.IMAGE)
        inputs = [asset]

    request = EmbedRequest(inputs=inputs, text=text, model=model, device=device)
    result = handle(request)[0]
    return Response(
        content=result.content_path.read_bytes(),
        media_type="application/octet-stream",
        headers={"X-Splat-Asset-Id": result.id},
    )
