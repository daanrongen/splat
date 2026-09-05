from fastapi.testclient import TestClient

from splat.domain.manifest import ManifestKind
from splat.domain.manifest_metadata import RasterMetadata
from splat.http.app import app
from splat.registry.wiring import get_asset_cache

client = TestClient(app)


def test_get_asset_returns_content_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    cache = get_asset_cache()
    asset = cache.put(
        "abc123",
        kind=ManifestKind.IMAGE,
        content_bytes=b"hello",
        ext="png",
        metadata=RasterMetadata(),
        parent_ids=[],
        created_by="test",
    )

    response = client.get(f"/assets/{asset.id}")

    assert response.status_code == 200
    assert response.content == b"hello"


def test_get_unknown_asset_returns_422(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))

    response = client.get("/assets/does-not-exist")

    assert response.status_code == 422
