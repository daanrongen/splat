import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from splat.domain.image_space import DepthMap, Shape3D
from splat.domain.value_objects import APPLE_ASCL, MIT
from splat.http.app import app

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


class FakeMeshBackend:
    name = "triposr"
    license = MIT

    def predict(self, image_path, **params) -> Shape3D:
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
        return Shape3D(vertices=vertices, faces=faces, metadata={"face_count": 2})


def _sample_png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (3, 2)).save(buf, format="PNG")
    return buf.getvalue()


def test_depth_returns_npy_bytes(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.depth.get_depth_backend", return_value=FakeDepthBackend())

    response = client.post(
        "/depth", files={"image": ("scene.png", _sample_png_bytes(), "image/png")}
    )

    assert response.status_code == 200, response.text
    assert len(response.content) > 0
    assert "X-Splat-Asset-Id" in response.headers


def test_mesh_returns_glb_bytes(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.mesh.get_mesh_backend", return_value=FakeMeshBackend())

    response = client.post(
        "/mesh", files={"image": ("scene.png", _sample_png_bytes(), "image/png")}
    )

    assert response.status_code == 200, response.text
    assert len(response.content) > 0
    assert "X-Splat-Asset-Id" in response.headers
