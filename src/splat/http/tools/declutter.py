import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, Response, UploadFile

from splat.handlers.tools.declutter import DeclutterRequest, handle

router = APIRouter()


@router.post("/declutter")
def declutter(
    input: UploadFile = File(...),
    to: str | None = Form(None),
    k: int = Form(16),
    std_ratio: float = Form(2.0),
) -> Response:
    suffix = Path(input.filename or "").suffix
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_path = tmp_path / f"input{suffix}"
        input_path.write_bytes(input.file.read())
        output_path = tmp_path / f"output.{to.lstrip('.') if to else suffix.lstrip('.')}"

        cloud = handle(
            DeclutterRequest(
                input_path=input_path, output_path=output_path, k=k, std_ratio=std_ratio
            )
        )
        content = output_path.read_bytes()

    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={"X-Splat-Point-Count": str(cloud.point_count)},
    )
