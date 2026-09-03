from pathlib import Path
from typing import Protocol

from splat.domain.asset import Asset, AssetKind


class AssetCache(Protocol):
    """Content-addressed store for the generative pipeline's Assets.
    `find`/`put` key on a caller-computed deterministic hash (see
    application/pipeline.py's compute_cache_key) so re-running the same
    stage+model+params+parents is a cache hit, not a recompute."""

    def find(self, asset_id: str) -> Asset | None: ...

    def get(self, asset_id: str) -> Asset: ...

    def put(
        self,
        asset_id: str,
        *,
        kind: AssetKind,
        content_bytes: bytes,
        ext: str,
        metadata: dict,
        parent_ids: list[str],
        created_by: str,
    ) -> Asset: ...

    def put_external(self, path: Path, *, kind: AssetKind) -> Asset:
        """Adopt an existing file (not produced by any stage) into the cache,
        e.g. a raw photo passed directly as a command's positional input."""
        ...
