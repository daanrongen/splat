import numpy as np

from splat.adapters.cache.filesystem import FilesystemManifestRepository
from splat.adapters.formats.image import encode_png
from splat.adapters.formats.ply import PlyWriter
from splat.application.pipeline import run_render
from splat.domain.manifest import ManifestKind
from splat.registry.wiring import get_writer


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


def test_run_render_creates_cached_image_asset(tmp_path, synthetic_cloud):
    cache = FilesystemManifestRepository(tmp_path / "cache")
    asset = _gaussian_asset(cache, tmp_path, synthetic_cloud)
    png_bytes = encode_png(np.zeros((4, 8, 3), dtype=np.uint8))
    backend = FakeRenderBackend(png_bytes)

    result = run_render(
        backend,
        cache,
        model_name="fake-render",
        input_asset=asset,
        params={"width": 8, "height": 4},
    )

    assert result.kind == ManifestKind.IMAGE
    assert result.metadata.output_width == 8
    assert result.metadata.output_height == 4
    assert result.parent_ids == [asset.id]
    assert result.created_by == "render:fake-render"


def test_run_render_reads_any_registered_splat_format(tmp_path, synthetic_cloud):
    """RENDER_CONTRACT accepts any splat_3d manifest, but the reader used to be
    a hardcoded PlyReader, so `splat render scene.splat` died on the header
    while `splat info` on the same file worked (gaps.md G11)."""
    cache = FilesystemManifestRepository(tmp_path / "cache")
    splat_path = tmp_path / "cloud.splat"
    get_writer(".splat").write(synthetic_cloud, splat_path)
    asset = cache.put_external(splat_path, kind=ManifestKind.GAUSSIAN_CLOUD)
    backend = FakeRenderBackend(encode_png(np.zeros((4, 8, 3), dtype=np.uint8)))

    result = run_render(
        backend, cache, model_name="fake-render", input_asset=asset, params={"width": 8}
    )

    assert result.kind == ManifestKind.IMAGE
    assert backend.calls == 1


def test_run_render_reuses_cache_for_same_inputs(tmp_path, synthetic_cloud):
    cache = FilesystemManifestRepository(tmp_path / "cache")
    asset = _gaussian_asset(cache, tmp_path, synthetic_cloud)
    png_bytes = encode_png(np.zeros((4, 8, 3), dtype=np.uint8))
    backend = FakeRenderBackend(png_bytes)

    first = run_render(
        backend,
        cache,
        model_name="fake-render",
        input_asset=asset,
        params={"width": 8, "height": 4},
    )
    second = run_render(
        backend,
        cache,
        model_name="fake-render",
        input_asset=asset,
        params={"width": 8, "height": 4},
    )

    assert first.id == second.id
    assert backend.calls == 1
