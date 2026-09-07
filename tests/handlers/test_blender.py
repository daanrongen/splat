import numpy as np
import pytest

from splat.adapters.formats.ply import PlyWriter
from splat.domain.errors import SplatDomainError
from splat.domain.manifest import ManifestKind
from splat.handlers.blender import BlenderRequest, handle
from splat.image_io import encode_png
from splat.registry.wiring import get_manifest_repository
from tests.image_helpers import write_sample_png


class FakeRenderBackend:
    name = "fake-render"

    def render(self, cloud, output_path, **params):
        output_path.write_bytes(encode_png(np.zeros((4, 8, 3), dtype=np.uint8)))


def _gaussian_asset(cache, tmp_path, synthetic_cloud):
    ply_path = tmp_path / "cloud.ply"
    PlyWriter().write(synthetic_cloud, ply_path)
    return cache.put_external(ply_path, kind=ManifestKind.GAUSSIAN_CLOUD)


def test_handle_creates_image_asset(mocker, tmp_path, monkeypatch, synthetic_cloud):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.blender.get_render_backend", return_value=FakeRenderBackend())
    cache = get_manifest_repository()
    asset = _gaussian_asset(cache, tmp_path, synthetic_cloud)

    results = handle(BlenderRequest(inputs=[asset]))

    assert len(results) == 1
    assert results[0].kind == ManifestKind.IMAGE
    assert results[0].parent_ids == [asset.id]
    assert results[0].created_by == "blender:fake-render"


def test_handle_rejects_non_gaussian_input(mocker, tmp_path, monkeypatch, synthetic_cloud):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.blender.get_render_backend", return_value=FakeRenderBackend())
    cache = get_manifest_repository()
    asset = cache.put_external(
        write_sample_png(tmp_path / "a.png", (2, 2)), kind=ManifestKind.IMAGE
    )

    with pytest.raises(SplatDomainError):
        handle(BlenderRequest(inputs=[asset]))
