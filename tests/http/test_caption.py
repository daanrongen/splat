from pathlib import Path

from fastapi.testclient import TestClient

from splat.domain.value_objects import MIT
from splat.http.app import app
from tests.image_helpers import sample_png_bytes

client = TestClient(app)


class FakeCaptionBackend:
    name = "fake-captioner"
    license = MIT

    def caption(
        self,
        image_path: Path,
        *,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **params,
    ) -> str:
        return f"{prompt}: small scene"


def test_caption_returns_text_bytes(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.caption.get_caption_backend", return_value=FakeCaptionBackend())

    response = client.post(
        "/caption",
        files={"image": ("scene.png", sample_png_bytes((3, 2)), "image/png")},
        data={"model": "fake-captioner", "prompt": "Look"},
    )

    assert response.status_code == 200, response.text
    assert response.content == b"Look: small scene"
    assert response.headers["content-type"] == "text/plain; charset=utf-8"
    assert "X-Splat-Asset-Id" in response.headers


def test_caption_unknown_model_returns_422(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))

    response = client.post(
        "/caption",
        files={"image": ("scene.png", sample_png_bytes((3, 2)), "image/png")},
        data={"model": "nope"},
    )

    assert response.status_code == 422
    assert "Unknown caption model" in response.json()["detail"]
