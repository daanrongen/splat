import dataclasses

from splat.domain.manifest import Manifest
from splat.handlers import manifest as manifest_handler


def _to_dict(manifest: Manifest) -> dict:
    return {
        "id": manifest.id,
        "kind": manifest.kind.value,
        "label": manifest.label,
        "created_by": manifest.created_by,
        "created_at": manifest.created_at,
        "content_size": manifest.content_size,
        "content_sha256": manifest.content_sha256,
        "parent_ids": manifest.parent_ids,
        "params": manifest.params,
        "metadata": dataclasses.asdict(manifest.metadata),
    }


def list_manifests(
    kind: str | None = None, created_by: str | None = None, limit: int = 50, offset: int = 0
) -> list[dict]:
    """List cached manifests, most recent first, optionally filtered by kind (e.g. "image")
    or a created_by substring (e.g. "diffuse")."""
    manifests = manifest_handler.list_manifests(
        kind=kind, created_by=created_by, limit=limit, offset=offset
    )
    return [_to_dict(m) for m in manifests]


def get(manifest_id: str) -> dict:
    """Get one cached manifest by id, with full typed metadata, stage params and children."""
    children = [child.id for child in manifest_handler.children(manifest_id)]
    return {**_to_dict(manifest_handler.get(manifest_id)), "children": children}


def delete(manifest_id: str, cascade: bool = False) -> dict:
    """Delete a cached manifest; refuses if others derive from it unless cascade."""
    manifest_handler.delete(manifest_id, cascade=cascade)
    return {"removed": manifest_id}


def label(manifest_id: str, text: str) -> dict:
    """Name a cached manifest so it is easy to find again; '' clears the label."""
    return _to_dict(manifest_handler.label(manifest_id, text))
