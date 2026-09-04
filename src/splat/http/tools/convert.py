import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, Response, UploadFile

from splat.handlers.tools.convert import ConvertRequest, handle

router = APIRouter()


@router.post("/convert")
def convert(
    input: UploadFile = File(...),
    to: str = Form(...),
    from_format: str | None = Form(None),
) -> Response:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_path = tmp_path / f"input{Path(input.filename or '').suffix}"
        input_path.write_bytes(input.file.read())
        output_path = tmp_path / f"output.{to.lstrip('.')}"

        result = handle(
            ConvertRequest(
                input_path=input_path,
                output_path=output_path,
                from_format=from_format,
            )
        )
        content = output_path.read_bytes()

    headers = {"X-Splat-Point-Count": str(result.cloud.point_count)}
    if result.warnings:
        headers["X-Splat-Warnings"] = "; ".join(result.warnings)
    return Response(content=content, media_type="application/octet-stream", headers=headers)
