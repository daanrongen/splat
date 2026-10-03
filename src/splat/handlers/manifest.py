"""CRUD over the local manifest cache. Unlike the model-backed stage
handlers, this isn't part of the SplatClient/SPLAT_URL contract: a manifest
already lives wherever it was produced (mirrored locally on every
manifest-producing remote call, see RemoteSplatClient._store_asset), so
`splat manifest ...` always reads the local cache directly — the same
boundary `tools.displace_height` draws for the same reason.
"""

from pathlib import Path

from splat.domain.errors import ManifestHasChildren, WrongManifestKind
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
    label: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> list[Manifest]:
    return _manifest_repository().list(
        kind=_parse_kind(kind), created_by=created_by, label=label, limit=limit, offset=offset
    )


def delete(manifest_id: str, *, cascade: bool = False) -> None:
    _manifest_repository().delete(manifest_id, cascade=cascade)


def label(manifest_id: str, text: str) -> Manifest:
    return _manifest_repository().set_label(manifest_id, text)


def children(manifest_id: str) -> list[Manifest]:
    return _manifest_repository().children(manifest_id)


def lineage(manifest_id: str) -> tuple[Manifest, list]:
    """(manifest, [lineage of each parent]); parents missing from the cache are skipped."""
    repository = _manifest_repository()

    def walk(manifest: Manifest) -> tuple[Manifest, list]:
        parents = [repository.find(pid) for pid in manifest.parent_ids]
        return manifest, [walk(parent) for parent in parents if parent is not None]

    return walk(repository.get(manifest_id))


def gc() -> list[Path]:
    return _manifest_repository().gc()


def clear(targets: list[Manifest]) -> None:
    """Deletes targets together, refusing if a manifest outside them derives from one."""
    repository = _manifest_repository()
    ids = {t.id for t in targets}
    outside = {c.id for t in targets for c in repository.children(t.id)} - ids
    if outside:
        raise ManifestHasChildren(
            f"{len(outside)} manifest(s) outside the selection derive from it; "
            "widen the filter or remove them first."
        )
    for target in targets:
        if repository.find(target.id) is not None:
            repository.delete(target.id, cascade=True)


def export(manifest_id: str, out_dir: Path) -> list[Path]:
    repository = _manifest_repository()
    repository.get(manifest_id)
    return repository.export(manifest_id, out_dir)
