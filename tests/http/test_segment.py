import numpy as np
from fastapi.testclient import TestClient

from splat.domain.image_space import Sticker
from splat.domain.value_objects import MIT
from splat.http.app import app
from tests.image_helpers import sample_png_bytes

client = TestClient(app)


class FakeSegmentationBackend:
    name = "fake-sam"
    license = MIT

    def segment(self, image_path, *, max_stickers=None, **params) -> list[Sticker]:
        rgba = np.zeros((2, 2, 4), dtype=np.uint8)
        return [Sticker(rgba=rgba, bbox=(0, 0, 2, 2), score=0.9, area=4)]


def _sample_png_bytes() -> bytes:
    return sample_png_bytes((2, 2))


def test_segment_returns_asset_manifest(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
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
