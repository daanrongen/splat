"""CRUD over the local manifest cache. Unlike the model-backed stage
handlers, this isn't part of the SplatClient/SPLAT_URL contract: a manifest
already lives wherever it was produced (mirrored locally on every
manifest-producing remote call, see RemoteSplatClient._store_asset), so
`splat manifest ...` always reads the local cache directly.
"""

from pathlib import Path

from splat.domain.errors import WrongManifestKind
from splat.domain.manifest import Manifest, ManifestKind


def _manifest_repository():
    from splat.adapters.cache.filesystem import FilesystemManifestRepository

    return FilesystemManifestRepository()


def _parse_kind(kind: str | None) -> ManifestKind | None:
    if kind is None:
        return None
    try:
        return ManifestKind(kind)
    except ValueError as exc:
        available = ", ".join(k.value for k in ManifestKind)
        raise WrongManifestKind(f"Unknown manifest kind {kind!r}. Available: {available}") from exc


def get(manifest_id: str) -> Manifest:
    return _manifest_repository().get(manifest_id)


def list_manifests(
    *,
    kind: str | None = None,
    created_by: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> list[Manifest]:
    return _manifest_repository().list(
        kind=_parse_kind(kind), created_by=created_by, limit=limit, offset=offset
    )


def delete(manifest_id: str) -> None:
    _manifest_repository().delete(manifest_id)


def export(manifest_id: str, out_dir: Path) -> list[Path]:
    repository = _manifest_repository()
    repository.get(manifest_id)
    return repository.export(manifest_id, out_dir)
