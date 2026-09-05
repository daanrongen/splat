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
        """Adopt an existing file (not produced by any stage) into the cache,
        e.g. a raw photo passed directly as a command's positional input."""
        ...

    def list(
        self,
        *,
        kind: ManifestKind | None = None,
        created_by: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Manifest]:
        """Every cached manifest, most recently created first. `created_by`
        matches by substring (e.g. "diffuse" finds every diffuse:* manifest)."""
        ...

    def delete(self, manifest_id: str) -> None:
        """Remove a manifest's content and metadata from the cache. Does not
        cascade to manifests that list it as a parent."""
        ...
