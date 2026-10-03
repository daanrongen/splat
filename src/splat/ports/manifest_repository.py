from __future__ import annotations

from pathlib import Path
from typing import Protocol

from splat.domain.manifest import Manifest, ManifestKind
from splat.domain.manifest_metadata import ManifestMetadata


class ManifestRepository(Protocol):
    """Content-addressed store for the generative pipeline's Manifests.
    `find`/`put` key on a caller-computed deterministic hash (see
    application/pipeline.py's compute_cache_key) so re-running the same
    stage+model+params+parents is a cache hit, not a recompute."""

    def find(self, manifest_id: str) -> Manifest | None: ...

    def get(self, manifest_id: str) -> Manifest: ...

    def put(
        self,
        manifest_id: str,
        *,
        kind: ManifestKind,
        content_bytes: bytes,
        ext: str,
        metadata: ManifestMetadata,
        params: dict | None = None,
        parent_ids: list[str],
        created_by: str,
    ) -> Manifest: ...

    def put_external(self, path: Path, *, kind: ManifestKind) -> Manifest:
        """Adopt an existing file into the cache, e.g. a raw photo passed as a
        command's input. A matching `.manifest.json` sidecar restores its identity."""
        ...

    def write_sidecar(self, manifest_id: str, path: Path) -> None:
        """Write `<path>.manifest.json` so the file carries its lineage out of the cache."""
        ...

    def export(self, manifest_id: str, out_dir: Path) -> list[Path]:
        """Copy a manifest and every cached ancestor, each with its sidecar, into out_dir."""
        ...

    def list(
        self,
        *,
        kind: ManifestKind | None = None,
        created_by: str | None = None,
        label: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Manifest]:
        """Every cached manifest, most recently created first. `created_by`
        matches by substring (e.g. "diffuse" finds every diffuse:* manifest)."""
        ...

    def set_label(self, manifest_id: str, label: str) -> Manifest: ...

    def children(self, manifest_id: str) -> list[Manifest]:
        """Manifests that list this one as a parent."""
        ...

    def delete(self, manifest_id: str, *, cascade: bool = False) -> None:
        """Remove a manifest's content and metadata. Refuses when other manifests
        derive from it unless `cascade`, which removes those too."""
        ...

    def gc(self) -> list[Path]:
        """Remove cache files no valid manifest owns; returns what was removed."""
        ...
