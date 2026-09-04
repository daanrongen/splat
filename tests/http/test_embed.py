from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from splat.domain.value_objects import MIT
from splat.http.app import app
from tests.image_helpers import sample_png_bytes

client = TestClient(app)


class FakeEmbeddingBackend:
    name = "fake-embedder"
    license = MIT

    def embed_image(self, image_path: Path, **params) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)

    def embed_text(self, text: str, **params) -> np.ndarray:
        return np.array([0.0, 1.0], dtype=np.float32)


def test_embed_image_returns_npy_bytes(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.embed.get_embedding_backend", return_value=FakeEmbeddingBackend())

    response = client.post(
        "/embed",
        files={"image": ("scene.png", sample_png_bytes((3, 2)), "image/png")},
        data={"model": "fake-embedder", "device": "cpu"},
    )

    assert response.status_code == 200, response.text
    assert np.load(BytesIO(response.content)).shape == (2,)
    assert response.headers["content-type"] == "application/octet-stream"
    assert "X-Splat-Asset-Id" in response.headers


def test_embed_text_returns_npy_bytes(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.embed.get_embedding_backend", return_value=FakeEmbeddingBackend())

    response = client.post(
        "/embed", data={"text": "red chair", "model": "fake-embedder", "device": "cpu"}
    )

    assert response.status_code == 200, response.text
    assert np.load(BytesIO(response.content)).shape == (2,)
    assert "X-Splat-Asset-Id" in response.headers


def test_embed_requires_image_or_text(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))

    response = client.post("/embed", data={"model": "fake-embedder"})

    assert response.status_code == 422
    assert "either image input or --text" in response.json()["detail"]
