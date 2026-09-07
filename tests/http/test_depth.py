import numpy as np
from fastapi.testclient import TestClient

from splat.domain.image_space import DepthMap
from splat.domain.value_objects import APPLE_ASCL
from splat.http.app import app
from tests.image_helpers import sample_png_bytes

client = TestClient(app)


class FakeDepthBackend:
    name = "depth-pro"
    license = APPLE_ASCL

    def estimate(self, image_path, **params) -> DepthMap:
        return DepthMap(
            depth=np.arange(6, dtype=np.float32).reshape(2, 3),
            focal_length_px=1234.5,
            field_of_view_deg=30.0,
            metadata={},
        )


def _sample_png_bytes() -> bytes:
    return sample_png_bytes((3, 2))


def test_depth_returns_npy_bytes(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.depth.get_depth_backend", return_value=FakeDepthBackend())

    response = client.post(
        "/depth", files={"image": ("scene.png", _sample_png_bytes(), "image/png")}
    )

    assert response.status_code == 200, response.text
    assert len(response.content) > 0
    assert "X-Splat-Asset-Id" in response.headers
