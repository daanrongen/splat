from fastapi import APIRouter, File, Form, Response, UploadFile

from splat.domain.manifest import ManifestKind
from splat.handlers.caption import DEFAULT_CAPTION_PROMPT, CaptionRequest, handle
from splat.http._files import saved_upload
from splat.registry.wiring import get_asset_cache

router = APIRouter()


@router.post("/caption")
def caption(
    image: UploadFile = File(...),
    model: str = Form("fastvlm-0.5b"),
    prompt: str = Form(DEFAULT_CAPTION_PROMPT),
    max_tokens: int = Form(80),
    temperature: float = Form(0.0),
    device: str = Form("auto"),
) -> Response:
    cache = get_asset_cache()
    with saved_upload(image) as path:
        asset = cache.put_external(path, kind=ManifestKind.IMAGE)
    result = handle(
        CaptionRequest(
            inputs=[asset],
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            device=device,
        )
    )[0]
    return Response(
        content=result.content_path.read_bytes(),
        media_type="text/plain; charset=utf-8",
        headers={"X-Splat-Asset-Id": result.id},
    )
