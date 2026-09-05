from pathlib import Path

import numpy as np

from splat.domain.image_space import DepthMap
from splat.domain.manifest import ManifestKind
from splat.domain.value_objects import APPLE_ASCL
from splat.handlers.depth import DepthRequest, handle
from splat.registry.wiring import get_manifest_repository
from tests.image_helpers import write_sample_png


class FakeDepthBackend:
    name = "depth-pro"
    license = APPLE_ASCL

    def estimate(self, image_path, **params) -> DepthMap:
        return DepthMap(
            depth=np.arange(6, dtype=np.float32).reshape(2, 3),
            focal_length_px=1234.5,
            field_of_view_deg=30.0,
            metadata={"source_model": "fake/depth"},
        )


def _sample_image(tmp_path: Path) -> Path:
    path = tmp_path / "scene.png"
    return write_sample_png(path, (3, 2))


def test_handle_estimates_depth_for_each_input(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.depth.get_depth_backend", return_value=FakeDepthBackend())
    cache = get_manifest_repository()
    asset = cache.put_external(_sample_image(tmp_path), kind=ManifestKind.IMAGE)

    results = handle(DepthRequest(inputs=[asset]))

    assert len(results) == 1
    assert results[0].kind == ManifestKind.DEPTH_MAP
    assert results[0].metadata.focal_length_px == 1234.5
    assert results[0].metadata.width == 3
    assert results[0].metadata.height == 2
    assert results[0].parent_ids == [asset.id]
