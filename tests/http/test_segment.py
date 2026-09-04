import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from splat.domain.image_space import Sticker
from splat.domain.value_objects import MIT
from splat.http.app import app

client = TestClient(app)


class FakeSegmentationBackend:
    name = "fake-sam"
    license = MIT

    def segment(self, image_path, *, max_stickers=None, **params) -> list[Sticker]:
        rgba = np.zeros((2, 2, 4), dtype=np.uint8)
        return [Sticker(rgba=rgba, bbox=(0, 0, 2, 2), score=0.9, area=4)]


def _sample_png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (2, 2)).save(buf, format="PNG")
    return buf.getvalue()


def test_segment_returns_asset_manifest(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch(
        "splat.handlers.segment.get_segmentation_backend",
        return_value=FakeSegmentationBackend(),
    )

    response = client.post(
        "/segment", files={"image": ("scene.png", _sample_png_bytes(), "image/png")}
    )

    assert response.status_code == 200, response.text
    manifest = response.json()
    assert len(manifest) == 1
    assert manifest[0]["kind"] == "sticker"
