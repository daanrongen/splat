import numpy as np
import pytest

from splat.adapters.formats.image import encode_png
from splat.adapters.formats.ply import PlyWriter
from splat.domain.errors import SplatDomainError
from splat.domain.manifest import ManifestKind
from splat.handlers.render import RenderRequest, handle
from splat.registry.wiring import get_manifest_repository
from tests.image_helpers import write_sample_png


class FakeRenderBackend:
    name = "fake-render"

    def __init__(self) -> None:
        self.params: dict = {}

    def render(self, cloud, output_path, **params):
        self.params = params
        output_path.write_bytes(encode_png(np.zeros((4, 8, 3), dtype=np.uint8)))


def _gaussian_asset(cache, tmp_path, synthetic_cloud):
    ply_path = tmp_path / "cloud.ply"
    PlyWriter().write(synthetic_cloud, ply_path)
    return cache.put_external(ply_path, kind=ManifestKind.GAUSSIAN_CLOUD)


def test_handle_creates_image_asset(mocker, tmp_path, monkeypatch, synthetic_cloud):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.render.get_render_backend", return_value=FakeRenderBackend())
    cache = get_manifest_repository()
    asset = _gaussian_asset(cache, tmp_path, synthetic_cloud)

    results = handle(RenderRequest(inputs=[asset], model="fake-render"))

    assert len(results) == 1
    assert results[0].kind == ManifestKind.IMAGE
    assert results[0].parent_ids == [asset.id]
    assert results[0].created_by == "render:fake-render"


def test_handle_forwards_camera_framing_to_the_backend(
    mocker, tmp_path, monkeypatch, synthetic_cloud
):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    backend = FakeRenderBackend()
    mocker.patch("splat.handlers.render.get_render_backend", return_value=backend)
    cache = get_manifest_repository()
    asset = _gaussian_asset(cache, tmp_path, synthetic_cloud)

    handle(
        RenderRequest(
            inputs=[asset], model="fake-render", azimuth=90.0, elevation=-10.0, look_at="0,1,2"
        )
    )

    assert backend.params["azimuth"] == 90.0
    assert backend.params["elevation"] == -10.0
    assert backend.params["look_at"] == "0,1,2"
    assert backend.params["distance"] is None


def test_handle_rejects_non_gaussian_input(mocker, tmp_path, monkeypatch, synthetic_cloud):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.render.get_render_backend", return_value=FakeRenderBackend())
    cache = get_manifest_repository()
    asset = cache.put_external(
        write_sample_png(tmp_path / "a.png", (2, 2)), kind=ManifestKind.IMAGE
    )

    with pytest.raises(SplatDomainError):
        handle(RenderRequest(inputs=[asset], model="fake-render"))
