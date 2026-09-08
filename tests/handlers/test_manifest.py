import pytest

from splat.domain.errors import WrongManifestKind
from splat.domain.manifest import ManifestKind
from splat.domain.manifest_metadata import RasterMetadata
from splat.handlers import manifest as manifest_handler
from splat.registry.wiring import get_manifest_repository


def _put_image(monkeypatch, tmp_path, manifest_id: str, created_by: str) -> None:
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    get_manifest_repository().put(
        manifest_id,
        kind=ManifestKind.IMAGE,
        content_bytes=manifest_id.encode(),
        ext="png",
        metadata=RasterMetadata(),
        parent_ids=[],
        created_by=created_by,
    )


def test_get_returns_the_manifest(tmp_path, monkeypatch):
    _put_image(monkeypatch, tmp_path, "a", "diffuse:sdxl")

    assert manifest_handler.get("a").created_by == "diffuse:sdxl"


def test_list_filters_by_kind_string(tmp_path, monkeypatch):
    _put_image(monkeypatch, tmp_path, "a", "diffuse:sdxl")

    rows = manifest_handler.list_manifests(kind="image")

    assert [m.id for m in rows] == ["a"]


def test_list_rejects_unknown_kind(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))

    with pytest.raises(WrongManifestKind, match="Unknown manifest kind"):
        manifest_handler.list_manifests(kind="not-a-kind")


def test_delete_removes_the_manifest(tmp_path, monkeypatch):
    _put_image(monkeypatch, tmp_path, "a", "diffuse:sdxl")

    manifest_handler.delete("a")

    assert manifest_handler.list_manifests() == []
