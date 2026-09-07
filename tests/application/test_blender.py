import numpy as np

from splat.adapters.cache.filesystem import FilesystemManifestRepository
from splat.adapters.formats.ply import PlyWriter
from splat.application.pipeline import run_blender
from splat.domain.manifest import ManifestKind
from splat.image_io import encode_png


class FakeRenderBackend:
    name = "fake-render"

    def __init__(self, png_bytes: bytes) -> None:
        self._png_bytes = png_bytes
        self.calls = 0

    def render(self, cloud, output_path, **params):
        self.calls += 1
        output_path.write_bytes(self._png_bytes)


def _gaussian_asset(cache, tmp_path, synthetic_cloud):
    ply_path = tmp_path / "cloud.ply"
    PlyWriter().write(synthetic_cloud, ply_path)
    return cache.put_external(ply_path, kind=ManifestKind.GAUSSIAN_CLOUD)


def test_run_blender_creates_cached_image_asset(tmp_path, synthetic_cloud):
    cache = FilesystemManifestRepository(tmp_path / "cache")
    asset = _gaussian_asset(cache, tmp_path, synthetic_cloud)
    png_bytes = encode_png(np.zeros((4, 8, 3), dtype=np.uint8))
    backend = FakeRenderBackend(png_bytes)

    result = run_blender(backend, cache, input_asset=asset, params={"width": 8, "height": 4})

    assert result.kind == ManifestKind.IMAGE
    assert result.metadata.output_width == 8
    assert result.metadata.output_height == 4
    assert result.parent_ids == [asset.id]
    assert result.created_by == "blender:fake-render"


def test_run_blender_reuses_cache_for_same_inputs(tmp_path, synthetic_cloud):
    cache = FilesystemManifestRepository(tmp_path / "cache")
    asset = _gaussian_asset(cache, tmp_path, synthetic_cloud)
    png_bytes = encode_png(np.zeros((4, 8, 3), dtype=np.uint8))
    backend = FakeRenderBackend(png_bytes)

    first = run_blender(backend, cache, input_asset=asset, params={"width": 8, "height": 4})
    second = run_blender(backend, cache, input_asset=asset, params={"width": 8, "height": 4})

    assert first.id == second.id
    assert backend.calls == 1
