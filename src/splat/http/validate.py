import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile

from splat.handlers.inspect import validate as validate_handler
from splat.http._schemas import ValidationResponse

router = APIRouter()


@router.post("/validate")
def validate(input: UploadFile = File(...), strict: bool = Form(False)) -> ValidationResponse:
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / f"input{Path(input.filename or '').suffix}"
        path.write_bytes(input.file.read())
        result = validate_handler(path, strict=strict)

    return ValidationResponse(
        valid=not result.issues, issues=result.issues, points=result.cloud.point_count
    )
