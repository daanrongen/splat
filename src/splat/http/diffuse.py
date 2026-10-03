from fastapi import APIRouter, File, Form, Response, UploadFile

from splat.domain.manifest import ManifestKind
from splat.handlers.diffuse import DiffuseRequest, handle
from splat.http._files import saved_upload
from splat.registry.wiring import get_manifest_repository

router = APIRouter()


@router.post("/diffuse")
def diffuse(
    prompt: str = Form(...),
    image: UploadFile | None = File(None),
    model: str = Form("sdxl-turbo-mlx"),
    negative_prompt: str = Form(""),
    steps: int | None = Form(None),
    strength: float | None = Form(None),
    seed: int | None = Form(None),
    width: int | None = Form(None),
    height: int | None = Form(None),
    device: str = Form("auto"),
) -> Response:
    inputs = []
    if image is not None:
        cache = get_manifest_repository()
        with saved_upload(image) as path:
            inputs = [cache.put_external(path, kind=ManifestKind.IMAGE)]

    result = handle(
        DiffuseRequest(
            prompt=prompt,
            inputs=inputs,
            model=model,
            negative_prompt=negative_prompt,
            steps=steps,
            strength=strength,
            seed=seed,
            width=width,
            height=height,
            device=device,
        )
    )
    headers = {"X-Splat-Asset-Id": result.asset.id}
    if result.license_warning:
        headers["X-Splat-License-Warning"] = result.license_warning
    return Response(
        content=result.asset.content_path.read_bytes(),
        media_type="image/png",
        headers=headers,
    )
