from pathlib import Path

import numpy as np

from splat.domain.image_space import Sticker
from splat.domain.manifest import ManifestKind
from splat.domain.value_objects import MIT
from splat.handlers.segment import SegmentRequest, handle
from splat.registry.wiring import get_asset_cache
from tests.image_helpers import write_sample_png


class FakeSegmentationBackend:
    name = "fake-sam"
    license = MIT

    def segment(self, image_path, *, max_stickers=None, **params) -> list[Sticker]:
        rgba = np.zeros((2, 2, 4), dtype=np.uint8)
        return [Sticker(rgba=rgba, bbox=(0, 0, 2, 2), score=0.9, area=4)]


def _sample_image(tmp_path: Path) -> Path:
    path = tmp_path / "scene.png"
    return write_sample_png(path, (2, 2))


def test_handle_fans_out_stickers(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch(
        "splat.handlers.segment.get_segmentation_backend",
        return_value=FakeSegmentationBackend(),
    )
    cache = get_asset_cache()
    asset = cache.put_external(_sample_image(tmp_path), kind=ManifestKind.IMAGE)

    result = handle(SegmentRequest(inputs=[asset], model="fake-sam"))

    assert len(result) == 1
    assert result[0].kind == ManifestKind.STICKER
    assert result[0].parent_ids == [asset.id]
