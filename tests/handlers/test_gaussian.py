import pytest

from splat.domain.asset import AssetKind
from splat.domain.errors import SplatDomainError
from splat.handlers.gaussian import GaussianRequest, handle
from splat.registry.wiring import get_asset_cache
from tests.image_helpers import write_sample_png


class FakeReconstructionBackend:
    name = "fake-recon"

    def __init__(self, cloud) -> None:
        self._cloud = cloud

    def reconstruct(self, images, *, device="auto", **params):
        return self._cloud

    def required_image_count(self) -> tuple[int, int | None]:
        return (2, None)


def test_handle_creates_gaussian_asset(mocker, tmp_path, monkeypatch, synthetic_cloud):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.gaussian.get_model_source", return_value=object())
    mocker.patch(
        "splat.handlers.gaussian.get_reconstruction_backend",
        return_value=FakeReconstructionBackend(synthetic_cloud),
    )
    cache = get_asset_cache()
    a = cache.put_external(write_sample_png(tmp_path / "a.png", (2, 2)), kind=AssetKind.IMAGE)
    b = cache.put_external(write_sample_png(tmp_path / "b.png", (2, 2)), kind=AssetKind.IMAGE)
    request = GaussianRequest(inputs=[a, b], model="fake-recon")

    results = handle(request)

    assert len(results) == 1
    assert results[0].kind == AssetKind.GAUSSIAN_CLOUD
    assert results[0].metadata["point_count"] == synthetic_cloud.point_count
    assert results[0].parent_ids == [a.id, b.id]
    assert results[0].created_by == "gaussian:fake-recon"


def test_handle_rejects_too_few_images(mocker, tmp_path, monkeypatch, synthetic_cloud):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.gaussian.get_model_source", return_value=object())
    mocker.patch(
        "splat.handlers.gaussian.get_reconstruction_backend",
        return_value=FakeReconstructionBackend(synthetic_cloud),
    )
    cache = get_asset_cache()
    asset = cache.put_external(write_sample_png(tmp_path / "a.png", (2, 2)), kind=AssetKind.IMAGE)
    request = GaussianRequest(inputs=[asset], model="fake-recon")

    with pytest.raises(SplatDomainError, match="requires between"):
        handle(request)
