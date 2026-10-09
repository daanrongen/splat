from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from splat.domain.errors import ManifestHasChildren, SplatDomainError
from splat.domain.manifest import Manifest, ManifestKind
from splat.domain.manifest_metadata import (
    KIND_METADATA_CLS,
    ManifestMetadata,
    RasterMetadata,
    StickerMetadata,
    check_contract,
    metadata_from_dict,
    metadata_to_dict,
)
from splat.paths import manifest_cache_dir

_RASTER_KINDS = (ManifestKind.IMAGE, ManifestKind.STICKER)

SIDECAR_SCHEMA = 1
_SIDECAR_KEYS = (
    "kind",
    "content_size",
    "content_sha256",
    "metadata",
    "params",
    "parent_ids",
    "created_by",
    "created_at",
    "label",
)


def sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".manifest.json")


def _read_sidecar(path: Path) -> dict[str, Any] | None:
    try:
        sidecar = json.loads(sidecar_path(path).read_text())
        ManifestKind(sidecar["kind"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    required = ("id", "content_sha256", "metadata", "parent_ids", "created_by")
    if not all(key in sidecar for key in required):
        return None
    return sidecar


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
        self._dir = cache_dir or manifest_cache_dir()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._sha_index_dir = self._dir / "sha256"
        if not self._sha_index_dir.exists():
            self._sha_index_dir.mkdir()
            self._build_sha_index()

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

    def _content_matches(self, content_path: Path, meta: dict[str, Any]) -> bool:
        try:
            stat = content_path.stat()
        except OSError:
            return False
        size, sha256 = meta.get("content_size"), meta.get("content_sha256")
        if size is not None and size != stat.st_size:
            return False
        # Unchanged size and mtime means unchanged bytes; skip rehashing large payloads.
        if sha256 is None or meta.get("content_mtime_ns") == stat.st_mtime_ns:
            return True
        return sha256 == hashlib.sha256(content_path.read_bytes()).hexdigest()

    def _manifest_from_meta(
        self, manifest_id: str, meta_path: Path, *, verify_content: bool = False
    ) -> Manifest | None:
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

        try:
            metadata = metadata_from_dict(kind, raw_metadata)
        except TypeError:
            return None

        content_path = self._dir / content_file
        content_size = meta.get("content_size")
        content_sha256 = meta.get("content_sha256")
        if verify_content:
            if not self._content_matches(content_path, meta):
                return None
        elif content_size is None:
            try:
                content_size = content_path.stat().st_size
            except OSError:
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
            content_size=content_size,
            content_sha256=content_sha256 or "",
            created_at=created_at,
            label=meta.get("label") if isinstance(meta.get("label"), str) else "",
        )

    def find(self, manifest_id: str) -> Manifest | None:
        meta_path = self._meta_path(manifest_id)
        if not meta_path.exists():
            return None
        return self._manifest_from_meta(manifest_id, meta_path, verify_content=True)

    def _sha_index_path(self, content_sha256: str) -> Path:
        return self._sha_index_dir / content_sha256

    def _find_by_content_sha256(self, content_sha256: str) -> Manifest | None:
        try:
            manifest_id = self._sha_index_path(content_sha256).read_text()
        except OSError:
            return None
        manifest = self.find(manifest_id)
        if manifest is None or manifest.content_sha256 != content_sha256:
            return None
        return manifest

    def _index_content(self, manifest: Manifest) -> None:
        """Points the digest at the first manifest holding those bytes, upgrading
        an external clone to one with real provenance when it appears."""
        current = self._find_by_content_sha256(manifest.content_sha256)
        if current is None or (
            current.created_by == "external" and manifest.created_by != "external"
        ):
            self._atomic_write_text(self._sha_index_path(manifest.content_sha256), manifest.id)

    def _build_sha_index(self) -> None:
        manifests = [m for m in self.list() if m.content_sha256]
        manifests.sort(key=lambda m: m.created_at)
        for manifest in manifests:
            self._index_content(manifest)

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
        if created_by != "external":
            check_contract(kind, metadata, content_bytes)
        content_file = f"{manifest_id}.{ext.lstrip('.')}"
        self._atomic_write_bytes(self._dir / content_file, content_bytes)
        content_mtime_ns = (self._dir / content_file).stat().st_mtime_ns

        existing_meta = self._load_meta(self._meta_path(manifest_id)) or {}
        created_at = existing_meta.get("created_at") or datetime.now(UTC).isoformat()

        meta = {
            "kind": kind.value,
            "content_file": content_file,
            "content_size": len(content_bytes),
            "content_sha256": hashlib.sha256(content_bytes).hexdigest(),
            "content_mtime_ns": content_mtime_ns,
            "metadata": metadata_to_dict(metadata),
            "params": params or {},
            "parent_ids": parent_ids,
            "created_by": created_by,
            "created_at": created_at,
            "label": existing_meta.get("label", ""),
        }
        self._atomic_write_text(
            self._meta_path(manifest_id),
            json.dumps(meta, sort_keys=True),
        )
        manifest = self.get(manifest_id)
        self._index_content(manifest)
        return manifest

    def _external_metadata(self, kind: ManifestKind, content_bytes: bytes) -> ManifestMetadata:
        if kind not in _RASTER_KINDS:
            return KIND_METADATA_CLS[kind]()
        from splat.adapters.formats.image import decode_rgb_or_rgba

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
        sidecar = _read_sidecar(path)
        if sidecar is not None and sidecar["content_sha256"] == content_sha256:
            return self._restore(path, content_bytes, sidecar)
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
            # A derived export (e.g. a format conversion) still points back at its source.
            parent_ids=[sidecar["id"]] if sidecar is not None else [],
            created_by="external",
        )

    def _restore(self, path: Path, content_bytes: bytes, sidecar: dict[str, Any]) -> Manifest:
        existing = self.find(sidecar["id"])
        if existing is not None:
            return existing
        kind = ManifestKind(sidecar["kind"])
        for parent_id in sidecar["parent_ids"]:
            for sibling in path.parent.glob(f"{parent_id}.*"):
                if _read_sidecar(sibling) is not None:
                    self.put_external(sibling, kind=kind)
        restored = self.put(
            sidecar["id"],
            kind=kind,
            content_bytes=content_bytes,
            ext=path.suffix.lstrip("."),
            metadata=metadata_from_dict(kind, sidecar["metadata"]),
            params=sidecar.get("params", {}),
            parent_ids=sidecar["parent_ids"],
            created_by=sidecar["created_by"],
        )
        if sidecar.get("label"):
            restored = self.set_label(restored.id, sidecar["label"])
        return restored

    def write_sidecar(self, manifest_id: str, path: Path) -> None:
        if os.environ.get("SPLAT_NO_MANIFEST"):
            return
        self._write_sidecar(manifest_id, path)

    def _write_sidecar(self, manifest_id: str, path: Path) -> None:
        meta = self._load_meta(self._meta_path(manifest_id)) or {}
        record = {key: value for key, value in meta.items() if key in _SIDECAR_KEYS}
        sidecar = {"schema": SIDECAR_SCHEMA, "id": manifest_id, **record}
        sidecar_path(path).write_text(json.dumps(sidecar, indent=2, sort_keys=True))

    def export(self, manifest_id: str, out_dir: Path) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        written, pending, seen = [], [manifest_id], set()
        while pending:
            current = pending.pop()
            manifest = self.find(current) if current not in seen else None
            seen.add(current)
            if manifest is None:
                continue
            target = out_dir / manifest.content_path.name
            shutil.copyfile(manifest.content_path, target)
            self._write_sidecar(manifest.id, target)
            written.append(target)
            pending.extend(manifest.parent_ids)
        return written

    def list(
        self,
        *,
        kind: ManifestKind | None = None,
        created_by: str | None = None,
        label: str | None = None,
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
            if label is not None and label not in manifest.label:
                continue
            manifests.append(manifest)

        manifests.sort(key=lambda m: m.created_at, reverse=True)
        end = None if limit is None else offset + limit
        return manifests[offset:end]

    def set_label(self, manifest_id: str, label: str) -> Manifest:
        self.get(manifest_id)
        meta_path = self._meta_path(manifest_id)
        meta = self._load_meta(meta_path) or {}
        self._atomic_write_text(meta_path, json.dumps({**meta, "label": label}, sort_keys=True))
        return self.get(manifest_id)

    def children(self, manifest_id: str) -> list[Manifest]:
        return [m for m in self.list() if manifest_id in m.parent_ids]

    def delete(self, manifest_id: str, *, cascade: bool = False) -> None:
        manifest = self.get(manifest_id)
        children = self.children(manifest_id)
        if children and not cascade:
            raise ManifestHasChildren(
                f"{manifest_id} has {len(children)} derived manifest(s); "
                "delete with cascade to remove them too."
            )
        for child in children:
            if self.find(child.id) is not None:
                self.delete(child.id, cascade=True)
        manifest.content_path.unlink(missing_ok=True)
        self._meta_path(manifest_id).unlink(missing_ok=True)
        if self._find_by_content_sha256(manifest.content_sha256) is None:
            self._sha_index_path(manifest.content_sha256).unlink(missing_ok=True)

    def gc(self) -> list[Path]:
        """Removes what no valid manifest owns: payloads without metadata, metadata
        whose payload is missing or corrupt, stale digest index entries."""
        valid = {m.id for m in self.list() if self.find(m.id) is not None}
        removed = []
        for path in self._dir.iterdir():
            if path.is_file() and path.name.split(".", 1)[0] not in valid:
                path.unlink()
                removed.append(path)
        for path in self._sha_index_dir.iterdir():
            if path.read_text() not in valid:
                path.unlink()
                removed.append(path)
        return removed
