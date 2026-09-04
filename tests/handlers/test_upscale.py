import numpy as np
import pytest

from splat.adapters.cache.filesystem import FilesystemAssetCache
from splat.domain.errors import SplatDomainError
from splat.domain.manifest import ManifestKind
from splat.domain.manifest_metadata import DepthMetadata
from splat.domain.value_objects import MIT
from splat.handlers.upscale import UpscaleRequest, handle
from tests.image_helpers import write_sample_png


class FakeUpscaleBackend:
    name = "fake-upscaler"
    license = MIT
    supported_factors = (2, 4)

    def upscale(self, image, *, factor: int, tile: int = 0, **params):
        return np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)


def test_handle_upscales_each_input(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.upscale.get_upscale_backend", return_value=FakeUpscaleBackend())
    cache = FilesystemAssetCache(tmp_path / "cache")
    asset = cache.put_external(
        write_sample_png(tmp_path / "image.png", (2, 2)),
        kind=ManifestKind.IMAGE,
    )

    results = handle(UpscaleRequest(inputs=[asset], model="fake-upscaler", factor=2, tile=8))

    assert len(results) == 1
    assert results[0].kind == ManifestKind.IMAGE
    assert results[0].parent_ids == [asset.id]
    assert results[0].created_by == "upscale:fake-upscaler"


def test_handle_rejects_non_image_input(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.upscale.get_upscale_backend", return_value=FakeUpscaleBackend())
    cache = FilesystemAssetCache(tmp_path / "cache")
    asset = cache.put(
        "depth",
        kind=ManifestKind.DEPTH_MAP,
        content_bytes=b"",
        ext="npy",
        metadata=DepthMetadata(),
        parent_ids=[],
        created_by="depth:test",
    )

    with pytest.raises(SplatDomainError, match="image or sticker"):
        handle(UpscaleRequest(inputs=[asset]))
