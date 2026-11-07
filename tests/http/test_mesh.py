import json

from fastapi.testclient import TestClient

from splat.adapters.formats.ply import PlyWriter
from splat.http.app import app

client = TestClient(app)


def _post_mesh(tmp_path, monkeypatch, synthetic_cloud, **data):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    ply_path = tmp_path / "cloud.ply"
    PlyWriter().write(synthetic_cloud, ply_path)
    return client.post(
        "/mesh",
        files={"cloud": ("cloud.ply", ply_path.read_bytes(), "application/octet-stream")},
        data={"resolution": "24", "opacity_threshold": "0", **data},
    )


def test_mesh_returns_the_mesh_with_its_metadata(tmp_path, monkeypatch, synthetic_cloud):
    response = _post_mesh(tmp_path, monkeypatch, synthetic_cloud)

    assert response.status_code == 200, response.text
    metadata = json.loads(response.headers["X-Splat-Metadata"])
    assert metadata["vertex_count"] > 0
    assert metadata["face_count"] > 0
    assert "X-Splat-Asset-Id" in response.headers


def test_mesh_rejects_a_model_that_does_not_fit_a_cloud(tmp_path, monkeypatch, synthetic_cloud):
    response = _post_mesh(tmp_path, monkeypatch, synthetic_cloud, model="heightfield")

    assert response.status_code == 422
    assert "heightfield needs a depth map" in response.json()["detail"]
