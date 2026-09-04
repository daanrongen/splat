import hashlib
import json

import pytest

from splat.adapters.cache.filesystem import AssetNotFound, FilesystemAssetCache
from splat.domain.asset import AssetKind


def _meta(*, content_file: str, content: bytes = b"asset-bytes") -> dict:
    return {
        "kind": AssetKind.IMAGE.value,
        "content_file": content_file,
        "content_size": len(content),
        "content_sha256": hashlib.sha256(content).hexdigest(),
        "metadata": {},
        "parent_ids": [],
        "created_by": "test",
    }


def test_find_returns_none_for_malformed_metadata(tmp_path):
    cache = FilesystemAssetCache(tmp_path)
    (tmp_path / "broken.meta.json").write_text("{not-json")

    assert cache.find("broken") is None

    with pytest.raises(AssetNotFound):
        cache.get("broken")


def test_find_returns_none_when_content_file_is_missing(tmp_path):
    cache = FilesystemAssetCache(tmp_path)
    (tmp_path / "missing.meta.json").write_text(json.dumps(_meta(content_file="missing.png")))

    assert cache.find("missing") is None


def test_find_ignores_incomplete_writes(tmp_path):
    cache = FilesystemAssetCache(tmp_path)
    (tmp_path / "content-only.png").write_bytes(b"asset-bytes")
    (tmp_path / "meta-only.meta.json.tmp").write_text(json.dumps(_meta(content_file="x.png")))

    assert cache.find("content-only") is None
    assert cache.find("meta-only") is None


def test_put_writes_validated_metadata(tmp_path):
    cache = FilesystemAssetCache(tmp_path)

    asset = cache.put(
        "asset",
        kind=AssetKind.IMAGE,
        content_bytes=b"asset-bytes",
        ext="png",
        metadata={"prompt": "test"},
        parent_ids=[],
        created_by="diffuse:test",
    )

    meta = json.loads((tmp_path / "asset.meta.json").read_text())
    assert meta["content_size"] == len(b"asset-bytes")
    assert meta["content_sha256"] == hashlib.sha256(b"asset-bytes").hexdigest()
    assert asset.content_path.read_bytes() == b"asset-bytes"


def test_find_returns_none_when_content_digest_mismatches(tmp_path):
    cache = FilesystemAssetCache(tmp_path)
    asset = cache.put(
        "asset",
        kind=AssetKind.IMAGE,
        content_bytes=b"asset-bytes",
        ext="png",
        metadata={},
        parent_ids=[],
        created_by="test",
    )
    asset.content_path.write_bytes(b"wrong-bytes")

    assert cache.find("asset") is None
