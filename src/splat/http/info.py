import tempfile
from pathlib import Path

from fastapi import APIRouter, File, UploadFile

from splat.handlers.inspect import info as info_handler
from splat.http._schemas import InfoResponse

router = APIRouter()


@router.post("/info")
def info(input: UploadFile = File(...)) -> InfoResponse:
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / f"input{Path(input.filename or '').suffix}"
        path.write_bytes(input.file.read())
        cloud = info_handler(path)

    return InfoResponse(
        format=cloud.metadata.source_format,
        points=cloud.point_count,
        sh_degree=cloud.sh_degree,
        bbox_min=cloud.means.min(axis=0).tolist(),
        bbox_max=cloud.means.max(axis=0).tolist(),
    )
