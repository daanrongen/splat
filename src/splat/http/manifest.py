from fastapi import APIRouter

from splat.handlers import manifest as manifest_handler
from splat.http._schemas import ManifestDetail, ManifestSummary

router = APIRouter()


@router.get("/manifests")
def list_manifests(
    kind: str | None = None, created_by: str | None = None, limit: int = 50, offset: int = 0
) -> list[ManifestSummary]:
    manifests = manifest_handler.list_manifests(
        kind=kind, created_by=created_by, limit=limit, offset=offset
    )
    return [ManifestSummary.from_manifest(m) for m in manifests]


@router.get("/manifests/{manifest_id}")
def get_manifest(manifest_id: str) -> ManifestDetail:
    return ManifestDetail.from_manifest(manifest_handler.get(manifest_id))


@router.delete("/manifests/{manifest_id}")
def delete_manifest(manifest_id: str) -> dict:
    manifest_handler.delete(manifest_id)
    return {"removed": manifest_id}
