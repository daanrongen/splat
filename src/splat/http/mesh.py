from fastapi import APIRouter, File, Form, Response, UploadFile

from splat.domain.manifest import ManifestKind
from splat.handlers.mesh import MeshRequest, handle
from splat.http._files import saved_upload
from splat.registry.wiring import get_asset_cache

router = APIRouter()


@router.post("/mesh")
def mesh(
    image: UploadFile = File(...),
    model: str = Form("triposr"),
    device: str = Form("auto"),
) -> Response:
    cache = get_asset_cache()
    with saved_upload(image) as path:
        asset = cache.put_external(path, kind=ManifestKind.IMAGE)
    result = handle(MeshRequest(inputs=[asset], model=model, device=device))[0]
    return Response(
        content=result.content_path.read_bytes(),
        media_type="model/gltf-binary",
        headers={"X-Splat-Asset-Id": result.id},
    )
