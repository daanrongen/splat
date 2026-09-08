import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from splat.adapters.formats.image import decode_rgb_or_rgba
from splat.domain.errors import SplatDomainError
from splat.domain.manifest import Manifest, ManifestKind
from splat.domain.manifest_metadata import (
    KIND_METADATA_CLS,
    MANIFEST_MARKER_EXT,
    ManifestMetadata,
    RasterMetadata,
    StickerMetadata,
    metadata_from_dict,
    metadata_to_dict,
)
from splat.paths import asset_cache_dir

_RASTER_KINDS = (ManifestKind.IMAGE, ManifestKind.STICKER)


class ManifestNotFound(SplatDomainError):
    pass


class FilesystemManifestRepository:
    """Cache: `<id>.<ext>` for the payload, `<id>.meta.json` for everything
    else. Both live flat in one directory — simple enough not to need
    sharding at this project's scale.

    `id` is an *invocation* key (`compute_cache_key` over stage/model/params/
    parents) for everything `put()` writes, which is what gives pipeline
    stages free memoization. `put_external()` addresses by content instead —
    it has no invocation to hash — and resolves against every existing
    manifest's `content_sha256` first, so a file that happens to hold the same
    bytes as an already-produced artifact reuses that artifact's id and
    provenance rather than minting a second, parentless manifest for it."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._dir = cache_dir or asset_cache_dir()
        self._dir.mkdir(parents=True, exist_ok=True)

    def _meta_path(self, manifest_id: str) -> Path:
        return self._dir / f"{manifest_id}.meta.json"

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

    def _manifest_from_meta(self, manifest_id: str, meta_path: Path) -> Manifest | None:
        meta = self._load_meta(meta_path)
        if meta is None:
            return None
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

        content_size = meta.get("content_size")
        if content_size is not None and content_size != len(content):
            return None
        content_sha256 = meta.get("content_sha256")
        if content_sha256 is not None and content_sha256 != hashlib.sha256(content).hexdigest():
            return None

        # Cache entries written before `created_at` existed fall back to the
        # metadata file's own mtime rather than losing the manifest.
        created_at = meta.get("created_at")
        if not isinstance(created_at, str) or not created_at:
            created_at = datetime.fromtimestamp(meta_path.stat().st_mtime, tz=UTC).isoformat()

        return Manifest(
            id=manifest_id,
            kind=kind,
            content_path=content_path,
            metadata=metadata,
            params=params,
            parent_ids=parent_ids,
            created_by=created_by,
            content_size=content_size if content_size is not None else len(content),
            content_sha256=content_sha256 or hashlib.sha256(content).hexdigest(),
            created_at=created_at,
        )

    def find(self, manifest_id: str) -> Manifest | None:
        meta_path = self._meta_path(manifest_id)
        if not meta_path.exists():
            return None
        return self._manifest_from_meta(manifest_id, meta_path)

    def _find_by_content_sha256(self, content_sha256: str) -> Manifest | None:
        """Resolves bytes to whichever manifest already holds them, preferring
        one with real provenance (`created_by != "external"`) over an earlier
        external clone, then the earliest match — so `list`/`get` stay stable
        across repeated lookups."""
        candidates = []
        for meta_path in self._dir.glob("*.meta.json"):
            meta = self._load_meta(meta_path)
            if meta is None or meta.get("content_sha256") != content_sha256:
                continue
            manifest_id = meta_path.name.removesuffix(".meta.json")
            manifest = self._manifest_from_meta(manifest_id, meta_path)
            if manifest is not None:
                candidates.append(manifest)
        if not candidates:
            return None
        candidates.sort(key=lambda m: (m.created_by == "external", m.created_at))
        return candidates[0]

    def get(self, manifest_id: str) -> Manifest:
        manifest = self.find(manifest_id)
        if manifest is None:
            raise ManifestNotFound(f"No cached manifest with id {manifest_id!r}")
        return manifest

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
    ) -> Manifest:
        content_file = f"{manifest_id}.{ext.lstrip('.')}"
        self._atomic_write_bytes(self._dir / content_file, content_bytes)

        existing_meta = self._load_meta(self._meta_path(manifest_id))
        created_at = (existing_meta or {}).get("created_at") or datetime.now(UTC).isoformat()

        meta = {
            "kind": kind.value,
            "content_file": content_file,
            "content_size": len(content_bytes),
            "content_sha256": hashlib.sha256(content_bytes).hexdigest(),
            "metadata": metadata_to_dict(metadata),
            "params": params or {},
            "parent_ids": parent_ids,
            "created_by": created_by,
            "created_at": created_at,
        }
        self._atomic_write_text(
            self._meta_path(manifest_id),
            json.dumps(meta, sort_keys=True),
        )
        return self.get(manifest_id)

    def _external_metadata(self, kind: ManifestKind, content_bytes: bytes) -> ManifestMetadata:
        if kind not in _RASTER_KINDS:
            return KIND_METADATA_CLS[kind]()
        image = decode_rgb_or_rgba(content_bytes)
        dims = {"output_width": image.shape[1], "output_height": image.shape[0]}
        if kind is ManifestKind.STICKER:
            return StickerMetadata(
                bbox=(0, 0, image.shape[1], image.shape[0]),
                score=1.0,
                area=int(image.shape[0] * image.shape[1]),
                width=image.shape[1],
                height=image.shape[0],
            )
        return RasterMetadata(**dims)

    def put_external(self, path: Path, *, kind: ManifestKind) -> Manifest:
        content_bytes = path.read_bytes()
        content_sha256 = hashlib.sha256(content_bytes).hexdigest()
        existing = self._find_by_content_sha256(content_sha256)
        if existing is not None:
            return existing
        return self.put(
            content_sha256[:16],
            kind=kind,
            content_bytes=content_bytes,
            ext=path.suffix.lstrip("."),
            metadata=self._external_metadata(kind, content_bytes),
            params={"source": "external", "original_name": path.name},
            parent_ids=[],
            created_by="external",
        )

    def list(
        self,
        *,
        kind: ManifestKind | None = None,
        created_by: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Manifest]:
        manifests = []
        for meta_path in self._dir.glob("*.meta.json"):
            manifest_id = meta_path.name.removesuffix(".meta.json")
            manifest = self._manifest_from_meta(manifest_id, meta_path)
            if manifest is None:
                continue
            if kind is not None and manifest.kind != kind:
                continue
            if created_by is not None and created_by not in manifest.created_by:
                continue
            manifests.append(manifest)

        manifests.sort(key=lambda m: m.created_at, reverse=True)
        end = None if limit is None else offset + limit
        return manifests[offset:end]

    def delete(self, manifest_id: str) -> None:
        manifest = self.get(manifest_id)
        manifest.content_path.unlink(missing_ok=True)
        self._meta_path(manifest_id).unlink(missing_ok=True)
