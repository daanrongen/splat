import pytest

from splat.adapters.cache.filesystem import FilesystemManifestRepository
from splat.domain.errors import ManifestHasChildren
from splat.domain.manifest import ManifestKind
from splat.domain.manifest_metadata import RasterMetadata


def _put(cache, manifest_id, parent_ids=()):
    return cache.put(
        manifest_id,
        kind=ManifestKind.IMAGE,
        content_bytes=manifest_id.encode(),
        ext="png",
        metadata=RasterMetadata(),
        parent_ids=list(parent_ids),
        created_by="test",
    )


def test_label_survives_a_cache_hit_rewrite(tmp_path):
    cache = FilesystemManifestRepository(tmp_path)
    _put(cache, "a")
    cache.set_label("a", "robot")

    _put(cache, "a")

    assert cache.get("a").label == "robot"
    assert [m.id for m in cache.list(label="rob")] == ["a"]


def test_delete_refuses_when_manifests_derive_from_it(tmp_path):
    cache = FilesystemManifestRepository(tmp_path)
    _put(cache, "base")
    _put(cache, "child", parent_ids=["base"])
    _put(cache, "grandchild", parent_ids=["child"])

    with pytest.raises(ManifestHasChildren):
        cache.delete("base")

    cache.delete("base", cascade=True)
    assert cache.list() == []


def test_gc_removes_files_no_valid_manifest_owns(tmp_path):
    cache = FilesystemManifestRepository(tmp_path)
    kept = _put(cache, "kept")
    broken = _put(cache, "broken")
    broken.content_path.unlink()
    (tmp_path / "orphan.png").write_bytes(b"orphan")

    removed = {p.name for p in cache.gc()}

    assert {"orphan.png", "broken.meta.json"} <= removed
    assert kept.content_path.exists()
    assert cache.get("kept").id == "kept"


def test_sidecar_carries_the_label(tmp_path):
    source = FilesystemManifestRepository(tmp_path / "a")
    _put(source, "robot")
    source.set_label("robot", "hero shot")
    out = tmp_path / "robot.png"
    out.write_bytes(b"robot")
    source.write_sidecar("robot", out)

    restored = FilesystemManifestRepository(tmp_path / "b").put_external(
        out, kind=ManifestKind.IMAGE
    )

    assert restored.label == "hero shot"
