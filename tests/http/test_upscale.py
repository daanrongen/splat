import numpy as np
from fastapi.testclient import TestClient

from splat.domain.value_objects import MIT
from splat.http.app import app
from splat.image_io import decode_rgb_or_rgba
from tests.image_helpers import sample_png_bytes

client = TestClient(app)


class FakeUpscaleBackend:
    name = "fake-upscaler"
    license = MIT
    supported_factors = (2, 4)

    def upscale(self, image, *, factor: int, tile: int = 0, **params):
        return np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)


def test_upscale_returns_png_bytes(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.upscale.get_upscale_backend", return_value=FakeUpscaleBackend())

    response = client.post(
        "/upscale",
        files={"image": ("scene.png", sample_png_bytes((3, 2)), "image/png")},
        data={"model": "fake-upscaler", "factor": "2"},
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/png"
    assert "X-Splat-Asset-Id" in response.headers
    assert decode_rgb_or_rgba(response.content).shape == (4, 6, 3)


def test_upscale_invalid_factor_returns_422(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.upscale.get_upscale_backend", return_value=FakeUpscaleBackend())

    response = client.post(
        "/upscale",
        files={"image": ("scene.png", sample_png_bytes((3, 2)), "image/png")},
        data={"model": "fake-upscaler", "factor": "3"},
    )

    assert response.status_code == 422
    assert "--factor 2, 4" in response.text
