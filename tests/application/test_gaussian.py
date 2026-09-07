from pathlib import Path

import numpy as np
import pytest

from splat.adapters.cache.filesystem import FilesystemManifestRepository
from splat.adapters.formats.ply import PlyReader
from splat.application.pipeline import run_gaussian
from splat.domain.manifest import ManifestKind
from splat.domain.value_objects import MIT
from tests.image_helpers import write_sample_png


class FakeReconstructionBackend:
    name = "fake-recon"
    license = MIT
    provides_metric_scale = False

    def __init__(self, cloud) -> None:
        self._cloud = cloud
        self.calls = 0

    def reconstruct(self, images, *, device="auto", **params):
        self.calls += 1
        return self._cloud

    def required_image_count(self) -> tuple[int, int | None]:
        return (2, None)


def _image_asset(cache: FilesystemManifestRepository, tmp_path: Path, name: str):
    return cache.put_external(write_sample_png(tmp_path / name, (2, 2)), kind=ManifestKind.IMAGE)


def test_run_gaussian_creates_cached_ply_asset(tmp_path, synthetic_cloud):
    cache = FilesystemManifestRepository(tmp_path / "cache")
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


def test_run_gaussian_normalizes_output_and_stores_a_canonical_y_up_cloud(
    tmp_path, synthetic_cloud
):
    """Backends report COLMAP (+Y down, scene at +Z). Storing that verbatim left
    every cloud upside down with the camera facing away in any Y-up consumer,
    and `up_axis` could not even describe it."""
    cache = FilesystemManifestRepository(tmp_path / "cache")
    a = _image_asset(cache, tmp_path, "a.png")
    b = _image_asset(cache, tmp_path, "b.png")
    backend = FakeReconstructionBackend(synthetic_cloud)
    backend._cloud.metadata.coordinate_convention = "colmap"

    result = run_gaussian(
        backend,
        cache,
        model_name="fake-recon",
        input_assets=[a, b],
        params={"device": "cpu"},
    )

    assert result.metadata.coordinate_convention == "opengl"
    assert result.metadata.up_axis == "y"
    cloud = PlyReader().read(result.content_path)
    radii = np.linalg.norm(cloud.means - np.median(cloud.means, axis=0), axis=1)
    assert np.median(radii) == pytest.approx(1.0, abs=1e-4)


def test_run_gaussian_reuses_cache_for_same_inputs(tmp_path, synthetic_cloud):
    cache = FilesystemManifestRepository(tmp_path / "cache")
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


class FakeMetricBackend(FakeReconstructionBackend):
    name = "fake-metric"
    provides_metric_scale = True

    def required_image_count(self) -> tuple[int, int | None]:
        return (1, 1)


def test_run_gaussian_preserves_metric_scale_when_the_backend_declares_it(
    tmp_path, synthetic_cloud
):
    """A metric backend's absolute scale is the one thing it uniquely provides;
    normalizing it to a median radius of 1.0 would throw that away."""
    cache = FilesystemManifestRepository(tmp_path / "cache")
    a = _image_asset(cache, tmp_path, "a.png")
    before = synthetic_cloud.means.copy()

    result = run_gaussian(
        FakeMetricBackend(synthetic_cloud),
        cache,
        model_name="fake-metric",
        input_assets=[a],
        params={"device": "cpu"},
    )

    assert result.metadata.metric_scale is True
    cloud = PlyReader().read(result.content_path)
    np.testing.assert_allclose(cloud.means, before, atol=1e-5)


def test_run_gaussian_marks_normalized_clouds_as_non_metric(tmp_path, synthetic_cloud):
    cache = FilesystemManifestRepository(tmp_path / "cache")
    a = _image_asset(cache, tmp_path, "a.png")
    b = _image_asset(cache, tmp_path, "b.png")

    result = run_gaussian(
        FakeReconstructionBackend(synthetic_cloud),
        cache,
        model_name="fake-recon",
        input_assets=[a, b],
        params={"device": "cpu"},
    )

    assert result.metadata.metric_scale is False


def test_run_gaussian_records_the_backend_license(tmp_path, synthetic_cloud):
    """Every reconstruction used to record license=None despite the descriptor
    naming one."""
    cache = FilesystemManifestRepository(tmp_path / "cache")
    a = _image_asset(cache, tmp_path, "a.png")
    b = _image_asset(cache, tmp_path, "b.png")

    result = run_gaussian(
        FakeReconstructionBackend(synthetic_cloud),
        cache,
        model_name="fake-recon",
        input_assets=[a, b],
        params={"device": "cpu"},
    )

    assert result.metadata.license == MIT
