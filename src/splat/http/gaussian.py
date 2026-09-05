import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, Response, UploadFile

from splat.domain.asset import AssetKind
from splat.handlers.gaussian import GaussianRequest, handle
from splat.registry.wiring import get_asset_cache, get_reader, get_writer

router = APIRouter()


@router.post("/gaussian")
def gaussian(
    images: list[UploadFile] = File(...),
    model: str = Form("mlx3d-capture"),
    device: str = Form("auto"),
    to: str = Form("ply"),
    quality: str = Form("fast"),
    iters: int | None = Form(None),
    max_dim: int | None = Form(None),
    sh_degree: int | None = Form(None),
    poses: str = Form("auto"),
    refine_poses: str = Form("auto"),
    low_memory: bool = Form(False),
    seed: int = Form(0),
) -> Response:
    cache = get_asset_cache()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_assets = []
        for i, image in enumerate(images):
            image_path = tmp_path / f"input_{i:03d}{Path(image.filename or '').suffix}"
            image_path.write_bytes(image.file.read())
            input_assets.append(cache.put_external(image_path, kind=AssetKind.IMAGE))
        output_path = tmp_path / f"output.{to.lstrip('.')}"

        result = handle(
            GaussianRequest(
                inputs=input_assets,
                model=model,
                device=device,
                quality=quality,
                iters=iters,
                max_dim=max_dim,
                sh_degree=sh_degree,
                poses=poses,
                refine_poses=refine_poses,
                low_memory=low_memory,
                seed=seed,
            )
        )[0]
        cloud = get_reader(result.content_path.suffix).read(result.content_path)
        writer = get_writer(output_path.suffix)
        warnings = writer.supports(cloud)
        writer.write(cloud, output_path)
        content = output_path.read_bytes()

    headers = {
        "X-Splat-Asset-Id": result.id,
        "X-Splat-Asset-Kind": result.kind.value,
        "X-Splat-Point-Count": str(result.metadata.get("point_count", cloud.point_count)),
    }
    if warnings:
        headers["X-Splat-Warnings"] = "; ".join(warnings)
    return Response(content=content, media_type="application/octet-stream", headers=headers)
