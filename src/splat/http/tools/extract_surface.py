import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, Response, UploadFile

from splat.handlers.tools.extract_surface import ExtractSurfaceRequest, handle

router = APIRouter()


@router.post("/extract-surface")
def extract_surface(
    input: UploadFile = File(...),
    to: str = Form("obj"),
    depth: int = Form(9),
    opacity_threshold: float = Form(0.1),
) -> Response:
    suffix = Path(input.filename or "").suffix
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_path = tmp_path / f"input{suffix}"
        input_path.write_bytes(input.file.read())
        output_path = tmp_path / f"output.{to.lstrip('.')}"

        result = handle(
            ExtractSurfaceRequest(
                input_path=input_path,
                output_path=output_path,
                format=to,
                depth=depth,
                opacity_threshold=opacity_threshold,
            )
        )
        content = output_path.read_bytes()

    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={
            "X-Splat-Point-Count": str(result.input_point_count),
            "X-Splat-Vertex-Count": str(result.vertex_count),
            "X-Splat-Face-Count": str(result.face_count),
        },
    )
