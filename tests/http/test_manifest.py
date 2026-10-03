from fastapi.testclient import TestClient

from splat.domain.manifest import ManifestKind
from splat.domain.manifest_metadata import RasterMetadata
from splat.http.app import app
from splat.registry.wiring import get_manifest_repository
from tests.image_helpers import tiny_png

client = TestClient(app)


def _put_image(monkeypatch, tmp_path, manifest_id: str, created_by: str = "diffuse:sdxl") -> None:
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    get_manifest_repository().put(
        manifest_id,
        kind=ManifestKind.IMAGE,
        content_bytes=tiny_png(manifest_id),
        ext="png",
        metadata=RasterMetadata(),
        parent_ids=[],
        created_by=created_by,
    )


def test_list_manifests_returns_summaries(tmp_path, monkeypatch):
    _put_image(monkeypatch, tmp_path, "a")

    response = client.get("/manifests")

    assert response.status_code == 200
    rows = response.json()
    assert len(rows) == 1
    assert rows[0]["id"] == "a"
    assert rows[0]["created_by"] == "diffuse:sdxl"
    assert "metadata" not in rows[0]


def test_list_manifests_filters_by_kind(tmp_path, monkeypatch):
    _put_image(monkeypatch, tmp_path, "a")

    response = client.get("/manifests", params={"kind": "depth_map"})

    assert response.status_code == 200
    assert response.json() == []


def test_list_manifests_rejects_unknown_kind(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))

    response = client.get("/manifests", params={"kind": "not-a-kind"})

    assert response.status_code == 422


def test_get_manifest_returns_full_detail(tmp_path, monkeypatch):
    _put_image(monkeypatch, tmp_path, "a")

    response = client.get("/manifests/a")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "a"
    assert "metadata" in body
    assert "params" in body


def test_get_unknown_manifest_returns_422(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))

    response = client.get("/manifests/does-not-exist")

    assert response.status_code == 422


def test_delete_manifest_removes_it(tmp_path, monkeypatch):
    _put_image(monkeypatch, tmp_path, "a")

    response = client.delete("/manifests/a")

    assert response.status_code == 200
    assert client.get("/manifests/a").status_code == 422
