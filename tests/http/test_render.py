import numpy as np
from fastapi.testclient import TestClient

from splat.adapters.formats.image import encode_png
from splat.adapters.formats.ply import PlyWriter
from splat.http.app import app

client = TestClient(app)

_FAKE_PNG = encode_png(np.zeros((4, 8, 3), dtype=np.uint8))


class FakeRenderBackend:
    name = "fake-render"

    def render(self, cloud, output_path, **params):
        output_path.write_bytes(_FAKE_PNG)


def test_render_returns_png(mocker, tmp_path, monkeypatch, synthetic_cloud):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.render.get_render_backend", return_value=FakeRenderBackend())
    ply_path = tmp_path / "cloud.ply"
    PlyWriter().write(synthetic_cloud, ply_path)

    response = client.post(
        "/render",
        files={"cloud": ("cloud.ply", ply_path.read_bytes(), "application/octet-stream")},
        data={"azimuth": "90"},
    )

    assert response.status_code == 200, response.text
    assert response.content == _FAKE_PNG
    assert response.headers["content-type"] == "image/png"
    assert "X-Splat-Asset-Id" in response.headers
