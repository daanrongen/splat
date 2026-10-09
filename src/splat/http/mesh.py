from fastapi import APIRouter, File, Form, Response, UploadFile

from splat.domain.manifest import ManifestKind
from splat.handlers.mesh import MeshRequest, handle
from splat.http._files import saved_upload
from splat.registry.wiring import get_manifest_repository

router = APIRouter()


@router.post("/mesh")
def mesh(
    cloud: UploadFile = File(...),
    format: str = Form("glb"),
    resolution: int = Form(192),
    opacity_threshold: float = Form(0.1),
) -> Response:
    cache = get_manifest_repository()
    with saved_upload(cloud) as path:
        asset = cache.put_external(path, kind=ManifestKind.GAUSSIAN_CLOUD)
    request = MeshRequest(
        inputs=[asset], format=format, resolution=resolution, opacity_threshold=opacity_threshold
    )
    result = handle(request)[0]
    return Response(
        content=result.content_path.read_bytes(),
        media_type="application/octet-stream",
        headers={"X-Splat-Asset-Id": result.id},
    )
