import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, Response, UploadFile

from splat.handlers.tools.compress import CompressRequest, handle

router = APIRouter()


@router.post("/compress")
def compress(
    input: UploadFile = File(...),
    profile: str = Form("web-delivery"),
    to: str | None = Form(None),
) -> Response:
    suffix = Path(input.filename or "").suffix
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_path = tmp_path / f"input{suffix}"
        input_path.write_bytes(input.file.read())
        output_path = tmp_path / f"output.{to.lstrip('.') if to else suffix.lstrip('.')}"

        cloud = handle(
            CompressRequest(input_path=input_path, output_path=output_path, profile=profile)
        )
        content = output_path.read_bytes()

    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={"X-Splat-Point-Count": str(cloud.point_count)},
    )
