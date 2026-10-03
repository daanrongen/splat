import hashlib
import json
import os

import pytest
from tests.image_helpers import write_sample_png

from splat.adapters.cache.filesystem import FilesystemManifestRepository, ManifestNotFound
from splat.domain.manifest import ManifestKind
from splat.domain.manifest_metadata import CaptionMetadata, RasterMetadata


def _meta(*, content_file: str, content: bytes = b"asset-bytes") -> dict:
    return {
        "kind": ManifestKind.IMAGE.value,
        "content_file": content_file,
        "content_size": len(content),
        "content_sha256": hashlib.sha256(content).hexdigest(),
        "metadata": {},
        "parent_ids": [],
        "created_by": "test",
    }


def test_find_returns_none_for_malformed_metadata(tmp_path):
    cache = FilesystemManifestRepository(tmp_path)
    (tmp_path / "broken.meta.json").write_text("{not-json")

    assert cache.find("broken") is None

    with pytest.raises(ManifestNotFound):
        cache.get("broken")


def test_find_returns_none_when_content_file_is_missing(tmp_path):
    cache = FilesystemManifestRepository(tmp_path)
    (tmp_path / "missing.meta.json").write_text(json.dumps(_meta(content_file="missing.png")))

    assert cache.find("missing") is None


def test_find_ignores_incomplete_writes(tmp_path):
    cache = FilesystemManifestRepository(tmp_path)
    (tmp_path / "content-only.png").write_bytes(b"asset-bytes")
    (tmp_path / "meta-only.meta.json.tmp").write_text(json.dumps(_meta(content_file="x.png")))

    assert cache.find("content-only") is None
    assert cache.find("meta-only") is None


def test_put_writes_validated_metadata(tmp_path):
    cache = FilesystemManifestRepository(tmp_path)

    asset = cache.put(
        "asset",
        kind=ManifestKind.IMAGE,
        content_bytes=b"asset-bytes",
        ext="png",
        metadata=RasterMetadata(),
        params={"prompt": "test"},
        parent_ids=[],
        created_by="diffuse:test",
    )

    meta = json.loads((tmp_path / "asset.meta.json").read_text())
    assert meta["content_size"] == len(b"asset-bytes")
    assert meta["content_sha256"] == hashlib.sha256(b"asset-bytes").hexdigest()
    assert asset.content_path.read_bytes() == b"asset-bytes"


def test_find_returns_none_when_content_digest_mismatches(tmp_path):
    cache = FilesystemManifestRepository(tmp_path)
    asset = cache.put(
        "asset",
        kind=ManifestKind.IMAGE,
        content_bytes=b"asset-bytes",
        ext="png",
        metadata=RasterMetadata(),
        parent_ids=[],
        created_by="test",
    )
    asset.content_path.write_bytes(b"wrong-bytes")

    assert cache.find("asset") is None


def test_put_populates_size_digest_and_timestamp_on_the_manifest(tmp_path):
    cache = FilesystemManifestRepository(tmp_path)

    asset = cache.put(
        "asset",
        kind=ManifestKind.IMAGE,
        content_bytes=b"asset-bytes",
        ext="png",
        metadata=RasterMetadata(),
        parent_ids=[],
        created_by="test",
    )

    assert asset.content_size == len(b"asset-bytes")
    assert asset.content_sha256 == hashlib.sha256(b"asset-bytes").hexdigest()
    assert asset.created_at

    reloaded = cache.get("asset")
    assert reloaded.created_at == asset.created_at


def test_find_falls_back_to_meta_mtime_when_created_at_is_missing(tmp_path):
    cache = FilesystemManifestRepository(tmp_path)
    content = b"asset-bytes"
    (tmp_path / "legacy.png").write_bytes(content)
    (tmp_path / "legacy.meta.json").write_text(json.dumps(_meta(content_file="legacy.png")))

    manifest = cache.get("legacy")

    assert manifest.created_at


def test_put_external_infers_image_dimensions(tmp_path):
    cache = FilesystemManifestRepository(tmp_path)
    source = write_sample_png(tmp_path / "photo.png", (4, 3))

    asset = cache.put_external(source, kind=ManifestKind.IMAGE)

    assert asset.metadata.output_width == 4
    assert asset.metadata.output_height == 3


def test_put_external_resolves_to_existing_manifest_with_same_bytes(tmp_path):
    cache = FilesystemManifestRepository(tmp_path)
    produced = cache.put(
        "6c21cef6c26e270e",
        kind=ManifestKind.IMAGE,
        content_bytes=b"same-bytes",
        ext="png",
        metadata=RasterMetadata(),
        params={"prompt": "a robot", "seed": 42},
        parent_ids=[],
        created_by="diffuse:sdxl-turbo-mlx",
    )
    copy_path = tmp_path / "downloaded.png"
    copy_path.write_bytes(b"same-bytes")

    resolved = cache.put_external(copy_path, kind=ManifestKind.IMAGE)

    assert resolved.id == produced.id
    assert resolved.created_by == "diffuse:sdxl-turbo-mlx"
    assert resolved.params == {"prompt": "a robot", "seed": 42}
    assert [m.id for m in cache.list()] == [produced.id]


def test_put_external_is_idempotent_for_the_same_bytes(tmp_path):
    cache = FilesystemManifestRepository(tmp_path)
    source = write_sample_png(tmp_path / "photo.png", (2, 2))

    first = cache.put_external(source, kind=ManifestKind.IMAGE)
    second = cache.put_external(source, kind=ManifestKind.IMAGE)

    assert first.id == second.id
    assert len(cache.list()) == 1


def test_put_external_prefers_pipeline_provenance_over_an_earlier_external_clone(tmp_path):
    cache = FilesystemManifestRepository(tmp_path)
    first_copy = write_sample_png(tmp_path / "first.png", (2, 2))
    content_bytes = first_copy.read_bytes()
    external = cache.put_external(first_copy, kind=ManifestKind.IMAGE)
    assert external.created_by == "external"

    produced = cache.put(
        "produced",
        kind=ManifestKind.IMAGE,
        content_bytes=content_bytes,
        ext="png",
        metadata=RasterMetadata(),
        parent_ids=[],
        created_by="diffuse:sdxl-turbo-mlx",
    )
    second_copy = tmp_path / "second.png"
    second_copy.write_bytes(content_bytes)

    resolved = cache.put_external(second_copy, kind=ManifestKind.IMAGE)

    assert resolved.id == produced.id


def test_find_does_not_rehash_unchanged_content(tmp_path, mocker):
    cache = FilesystemManifestRepository(tmp_path)
    cache.put(
        "asset",
        kind=ManifestKind.IMAGE,
        content_bytes=b"asset-bytes",
        ext="png",
        metadata=RasterMetadata(),
        parent_ids=[],
        created_by="test",
    )
    sha256 = mocker.spy(hashlib, "sha256")

    assert cache.get("asset").id == "asset"
    sha256.assert_not_called()


def test_find_rehashes_when_mtime_changes(tmp_path):
    cache = FilesystemManifestRepository(tmp_path)
    asset = cache.put(
        "asset",
        kind=ManifestKind.IMAGE,
        content_bytes=b"asset-bytes",
        ext="png",
        metadata=RasterMetadata(),
        parent_ids=[],
        created_by="test",
    )
    asset.content_path.write_bytes(b"asset-bytez")
    os.utime(asset.content_path, ns=(1, 1))

    assert cache.find("asset") is None


def test_put_external_resolves_bytes_cached_before_the_index_existed(tmp_path):
    content = b"same-bytes"
    (tmp_path / "legacy.png").write_bytes(content)
    meta = {**_meta(content_file="legacy.png", content=content), "created_by": "diffuse:x"}
    (tmp_path / "legacy.meta.json").write_text(json.dumps(meta))
    copy_path = tmp_path / "copy.png"
    copy_path.write_bytes(content)

    resolved = FilesystemManifestRepository(tmp_path).put_external(
        copy_path, kind=ManifestKind.IMAGE
    )

    assert resolved.id == "legacy"


def test_put_external_after_delete_creates_a_fresh_manifest(tmp_path):
    cache = FilesystemManifestRepository(tmp_path)
    source = write_sample_png(tmp_path / "photo.png", (2, 2))
    first = cache.put_external(source, kind=ManifestKind.IMAGE)
    cache.delete(first.id)

    second = cache.put_external(source, kind=ManifestKind.IMAGE)

    assert second.content_path.exists()
    assert [m.id for m in cache.list()] == [second.id]


def test_list_filters_by_kind_and_created_by_most_recent_first(tmp_path):
    cache = FilesystemManifestRepository(tmp_path)
    cache.put(
        "a",
        kind=ManifestKind.IMAGE,
        content_bytes=b"a",
        ext="png",
        metadata=RasterMetadata(),
        parent_ids=[],
        created_by="diffuse:sdxl",
    )
    cache.put(
        "b",
        kind=ManifestKind.IMAGE,
        content_bytes=b"b",
        ext="png",
        metadata=RasterMetadata(),
        parent_ids=[],
        created_by="upscale:realesrgan",
    )
    cache.put(
        "c",
        kind=ManifestKind.CAPTION,
        content_bytes=b"c",
        ext="txt",
        metadata=CaptionMetadata(text_length=1),
        parent_ids=[],
        created_by="diffuse:sdxl",
    )

    assert {m.id for m in cache.list()} == {"a", "b", "c"}
    assert [m.id for m in cache.list(kind=ManifestKind.IMAGE)] == ["b", "a"]
    assert {m.id for m in cache.list(created_by="diffuse")} == {"a", "c"}
    assert [m.id for m in cache.list(limit=1)] == ["c"]
    assert [m.id for m in cache.list(limit=1, offset=1)] == ["b"]


def test_delete_removes_content_and_metadata(tmp_path):
    cache = FilesystemManifestRepository(tmp_path)
    asset = cache.put(
        "asset",
        kind=ManifestKind.IMAGE,
        content_bytes=b"asset-bytes",
        ext="png",
        metadata=RasterMetadata(),
        parent_ids=[],
        created_by="test",
    )

    cache.delete("asset")

    assert cache.find("asset") is None
    assert not asset.content_path.exists()
    assert cache.list() == []


def test_delete_raises_for_unknown_id(tmp_path):
    cache = FilesystemManifestRepository(tmp_path)

    with pytest.raises(ManifestNotFound):
        cache.delete("does-not-exist")
