import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, Response, UploadFile

from splat.domain.manifest import ManifestKind
from splat.handlers.export import ExportRequest, handle
from splat.http._files import saved_upload
from splat.registry.wiring import get_manifest_repository

router = APIRouter()


@router.post("/export")
def export(
    input: UploadFile = File(...),
    to: str = Form(...),
    profile: str | None = Form(None),
    pruning: str = Form("threshold"),
    target_count: int | None = Form(None),
) -> Response:
    with saved_upload(input) as path:
        asset = get_manifest_repository().put_external(path, kind=ManifestKind.GAUSSIAN_CLOUD)
    with tempfile.TemporaryDirectory() as tmp_dir:
        output = Path(tmp_dir) / f"export.{to.lstrip('.')}"
        result = handle(
            ExportRequest(
                input=asset,
                output_path=output,
                profile=profile,
                pruning=pruning,
                target_count=target_count,
            )
        )
        content = output.read_bytes()
    headers = {"X-Splat-Asset-Id": asset.id}
    if result.point_count is not None:
        headers["X-Splat-Point-Count"] = str(result.point_count)
    if result.warnings:
        headers["X-Splat-Warnings"] = "; ".join(result.warnings)
    return Response(content=content, media_type="application/octet-stream", headers=headers)
