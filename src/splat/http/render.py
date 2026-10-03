from fastapi import APIRouter, File, Form, Response, UploadFile

from splat.domain.manifest import ManifestKind
from splat.handlers.render import RenderRequest, handle
from splat.http._files import saved_upload
from splat.registry.wiring import get_manifest_repository

router = APIRouter()


@router.post("/render")
def render(
    cloud: UploadFile = File(...),
    width: int = Form(1280),
    height: int = Form(720),
    samples: int = Form(32),
    engine: str = Form("cycles"),
    background: str = Form("black"),
    azimuth: float | None = Form(None),
    elevation: float | None = Form(None),
    distance: float | None = Form(None),
    fov: float | None = Form(None),
    look_at: str | None = Form(None),
    view: int | None = Form(None),
) -> Response:
    cache = get_manifest_repository()
    with saved_upload(cloud) as path:
        asset = cache.put_external(path, kind=ManifestKind.GAUSSIAN_CLOUD)
    request = RenderRequest(
        inputs=[asset],
        width=width,
        height=height,
        samples=samples,
        engine=engine,
        background=background,
        azimuth=azimuth,
        elevation=elevation,
        distance=distance,
        fov=fov,
        look_at=look_at,
        view=view,
    )
    result = handle(request)[0]
    return Response(
        content=result.content_path.read_bytes(),
        media_type="image/png",
        headers={"X-Splat-Asset-Id": result.id},
    )
