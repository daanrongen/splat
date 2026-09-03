import hashlib
import json
from pathlib import Path

from splat.domain.asset import Asset, AssetKind
from splat.domain.errors import SplatDomainError
from splat.paths import asset_cache_dir


class AssetNotFound(SplatDomainError):
    pass


class FilesystemAssetCache:
    """Content-addressed cache: `<id>.<ext>` for the payload, `<id>.meta.json`
    for everything else. Both live flat in one directory — simple enough not
    to need sharding at this project's scale."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._dir = cache_dir or asset_cache_dir()
        self._dir.mkdir(parents=True, exist_ok=True)

    def _meta_path(self, asset_id: str) -> Path:
        return self._dir / f"{asset_id}.meta.json"

    def find(self, asset_id: str) -> Asset | None:
        meta_path = self._meta_path(asset_id)
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text())
        return Asset(
            id=asset_id,
            kind=AssetKind(meta["kind"]),
            content_path=self._dir / meta["content_file"],
            metadata=meta["metadata"],
            parent_ids=meta["parent_ids"],
            created_by=meta["created_by"],
        )

    def get(self, asset_id: str) -> Asset:
        asset = self.find(asset_id)
        if asset is None:
            raise AssetNotFound(f"No cached asset with id {asset_id!r}")
        return asset

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
    ) -> Asset:
        content_file = f"{asset_id}.{ext.lstrip('.')}"
        (self._dir / content_file).write_bytes(content_bytes)
        self._meta_path(asset_id).write_text(
            json.dumps(
                {
                    "kind": kind.value,
                    "content_file": content_file,
                    "metadata": metadata,
                    "parent_ids": parent_ids,
                    "created_by": created_by,
                }
            )
        )
        return self.get(asset_id)

    def put_external(self, path: Path, *, kind: AssetKind) -> Asset:
        content_bytes = path.read_bytes()
        asset_id = hashlib.sha256(content_bytes).hexdigest()[:16]
        existing = self.find(asset_id)
        if existing is not None:
            return existing
        return self.put(
            asset_id,
            kind=kind,
            content_bytes=content_bytes,
            ext=path.suffix.lstrip("."),
            metadata={"source": "external", "original_name": path.name},
            parent_ids=[],
            created_by="external",
        )
