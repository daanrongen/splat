import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, Response, UploadFile

from splat.handlers.gaussian import GaussianRequest, handle

router = APIRouter()


@router.post("/gaussian")
def gaussian(
    images: list[UploadFile] = File(...),
    model: str = Form("mvsplat"),
    device: str = Form("auto"),
    to: str = Form("ply"),
) -> Response:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_paths = []
        for i, image in enumerate(images):
            image_path = tmp_path / f"input_{i:03d}{Path(image.filename or '').suffix}"
            image_path.write_bytes(image.file.read())
            input_paths.append(image_path)
        output_path = tmp_path / f"output.{to.lstrip('.')}"

        result = handle(
            GaussianRequest(inputs=input_paths, output_path=output_path, model=model, device=device)
        )
        content = output_path.read_bytes()

    headers = {"X-Splat-Point-Count": str(result.cloud.point_count)}
    if result.warnings:
        headers["X-Splat-Warnings"] = "; ".join(result.warnings)
    return Response(content=content, media_type="application/octet-stream", headers=headers)
