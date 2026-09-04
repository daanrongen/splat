from fastapi import APIRouter, Response

from splat.handlers.diffuse import DiffuseRequest, handle
from splat.http._schemas import DiffuseBody

router = APIRouter()


@router.post("/diffuse")
def diffuse(body: DiffuseBody) -> Response:
    result = handle(
        DiffuseRequest(
            prompt=body.prompt,
            model=body.model,
            negative_prompt=body.negative_prompt,
            steps=body.steps,
            seed=body.seed,
            device=body.device,
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
