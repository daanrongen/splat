from pathlib import Path

from splat.adapters.cache.filesystem import FilesystemAssetCache
from splat.adapters.formats.ply import PlyReader
from splat.application.pipeline import run_gaussian
from splat.domain.manifest import ManifestKind
from splat.domain.value_objects import MIT
from tests.image_helpers import write_sample_png


class FakeReconstructionBackend:
    name = "fake-recon"
    license = MIT

    def __init__(self, cloud) -> None:
        self._cloud = cloud
        self.calls = 0

    def reconstruct(self, images, *, device="auto", **params):
        self.calls += 1
        return self._cloud

    def required_image_count(self) -> tuple[int, int | None]:
        return (2, None)


def _image_asset(cache: FilesystemAssetCache, tmp_path: Path, name: str):
    return cache.put_external(write_sample_png(tmp_path / name, (2, 2)), kind=ManifestKind.IMAGE)


def test_run_gaussian_creates_cached_ply_asset(tmp_path, synthetic_cloud):
    cache = FilesystemAssetCache(tmp_path / "cache")
    a = _image_asset(cache, tmp_path, "a.png")
    b = _image_asset(cache, tmp_path, "b.png")

    result = run_gaussian(
        FakeReconstructionBackend(synthetic_cloud),
        cache,
        model_name="fake-recon",
        input_assets=[a, b],
        params={"device": "cpu"},
    )

    assert result.kind == ManifestKind.GAUSSIAN_CLOUD
    assert result.content_path.suffix == ".ply"
    assert result.parent_ids == [a.id, b.id]
    assert result.created_by == "gaussian:fake-recon"
    assert result.metadata.point_count == synthetic_cloud.point_count
    assert PlyReader().read(result.content_path).point_count == synthetic_cloud.point_count


def test_run_gaussian_reuses_cache_for_same_inputs(tmp_path, synthetic_cloud):
    cache = FilesystemAssetCache(tmp_path / "cache")
    a = _image_asset(cache, tmp_path, "a.png")
    b = _image_asset(cache, tmp_path, "b.png")
    backend = FakeReconstructionBackend(synthetic_cloud)

    first = run_gaussian(
        backend,
        cache,
        model_name="fake-recon",
        input_assets=[a, b],
        params={"device": "cpu"},
    )
    second = run_gaussian(
        backend,
        cache,
        model_name="fake-recon",
        input_assets=[a, b],
        params={"device": "cpu"},
    )

    assert first.id == second.id
    assert backend.calls == 1
