import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from splat.domain.errors import SplatDomainError
from splat.domain.manifest import Manifest, ManifestKind
from splat.domain.manifest_metadata import (
    KIND_METADATA_CLS,
    MANIFEST_MARKER_EXT,
    ManifestMetadata,
    metadata_from_dict,
    metadata_to_dict,
)
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

    def _atomic_write_bytes(self, path: Path, data: bytes) -> None:
        with tempfile.NamedTemporaryFile(dir=self._dir, delete=False) as tmp:
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        tmp_path.replace(path)

    def _atomic_write_text(self, path: Path, data: str) -> None:
        self._atomic_write_bytes(path, data.encode("utf-8"))

    def _load_meta(self, meta_path: Path) -> dict[str, Any] | None:
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        return meta if isinstance(meta, dict) else None

    def _asset_from_meta(self, asset_id: str, meta: dict[str, Any]) -> Manifest | None:
        try:
            content_file = meta["content_file"]
            kind = ManifestKind(meta["kind"])
            raw_metadata = meta["metadata"]
            params = meta.get("params", {})
            parent_ids = meta["parent_ids"]
            created_by = meta["created_by"]
        except (KeyError, TypeError, ValueError):
            return None

        if not isinstance(content_file, str) or Path(content_file).name != content_file:
            return None
        if not isinstance(raw_metadata, dict) or not isinstance(params, dict):
            return None
        if not isinstance(parent_ids, list) or not all(
            isinstance(parent_id, str) for parent_id in parent_ids
        ):
            return None
        if not isinstance(created_by, str):
            return None

        is_marker = Path(content_file).suffix == f".{MANIFEST_MARKER_EXT}"
        try:
            metadata = metadata_from_dict(kind, raw_metadata, is_manifest_marker=is_marker)
        except TypeError:
            return None

        content_path = self._dir / content_file
        try:
            content = content_path.read_bytes()
        except OSError:
            return None

        if meta.get("content_size") is not None and meta["content_size"] != len(content):
            return None
        digest = meta.get("content_sha256")
        if digest is not None and digest != hashlib.sha256(content).hexdigest():
            return None

        return Manifest(
            id=asset_id,
            kind=kind,
            content_path=content_path,
            metadata=metadata,
            params=params,
            parent_ids=parent_ids,
            created_by=created_by,
        )

    def find(self, asset_id: str) -> Manifest | None:
        meta_path = self._meta_path(asset_id)
        if not meta_path.exists():
            return None
        meta = self._load_meta(meta_path)
        if meta is None:
            return None
        return self._asset_from_meta(asset_id, meta)

    def get(self, asset_id: str) -> Manifest:
        asset = self.find(asset_id)
        if asset is None:
            raise AssetNotFound(f"No cached asset with id {asset_id!r}")
        return asset

    def put(
        self,
        asset_id: str,
        *,
        kind: ManifestKind,
        content_bytes: bytes,
        ext: str,
        metadata: ManifestMetadata,
        params: dict | None = None,
        parent_ids: list[str],
        created_by: str,
    ) -> Manifest:
        content_file = f"{asset_id}.{ext.lstrip('.')}"
        self._atomic_write_bytes(self._dir / content_file, content_bytes)
        meta = {
            "kind": kind.value,
            "content_file": content_file,
            "content_size": len(content_bytes),
            "content_sha256": hashlib.sha256(content_bytes).hexdigest(),
            "metadata": metadata_to_dict(metadata),
            "params": params or {},
            "parent_ids": parent_ids,
            "created_by": created_by,
        }
        self._atomic_write_text(
            self._meta_path(asset_id),
            json.dumps(meta, sort_keys=True),
        )
        return self.get(asset_id)

    def put_external(self, path: Path, *, kind: ManifestKind) -> Manifest:
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
            metadata=KIND_METADATA_CLS[kind](),
            params={"source": "external", "original_name": path.name},
            parent_ids=[],
            created_by="external",
        )
