import pytest

from splat.domain.errors import SplatDomainError
from splat.domain.manifest import ManifestKind
from splat.domain.value_objects import MIT
from splat.handlers.gaussian import (
    GaussianRequest,
    _input_hint,
    _single_image_models,
    handle,
)
from splat.registry.wiring import get_manifest_repository
from tests.image_helpers import write_sample_png


class FakeReconstructionBackend:
    name = "fake-recon"
    license = MIT
    provides_metric_scale = False

    def __init__(self, cloud) -> None:
        self._cloud = cloud

    def reconstruct(self, images, *, device="auto", **params):
        return self._cloud

    def required_image_count(self) -> tuple[int, int | None]:
        return (2, None)


def test_handle_creates_gaussian_asset(mocker, tmp_path, monkeypatch, synthetic_cloud):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.gaussian.get_model_source", return_value=object())
    mocker.patch(
        "splat.handlers.gaussian.get_reconstruction_backend",
        return_value=FakeReconstructionBackend(synthetic_cloud),
    )
    cache = get_manifest_repository()
    a = cache.put_external(write_sample_png(tmp_path / "a.png", (2, 2)), kind=ManifestKind.IMAGE)
    b = cache.put_external(write_sample_png(tmp_path / "b.png", (2, 2)), kind=ManifestKind.IMAGE)
    request = GaussianRequest(inputs=[a, b], model="fake-recon")

    results = handle(request)

    assert len(results) == 1
    assert results[0].kind == ManifestKind.GAUSSIAN_CLOUD
    assert results[0].metadata.point_count == synthetic_cloud.point_count
    assert results[0].parent_ids == [a.id, b.id]
    assert results[0].created_by == "gaussian:fake-recon"


def test_handle_rejects_too_few_images(mocker, tmp_path, monkeypatch, synthetic_cloud):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.gaussian.get_model_source", return_value=object())
    mocker.patch(
        "splat.handlers.gaussian.get_reconstruction_backend",
        return_value=FakeReconstructionBackend(synthetic_cloud),
    )
    cache = get_manifest_repository()
    asset = cache.put_external(
        write_sample_png(tmp_path / "a.png", (2, 2)), kind=ManifestKind.IMAGE
    )
    request = GaussianRequest(inputs=[asset], model="fake-recon")

    with pytest.raises(SplatDomainError, match="requires at least 2"):
        handle(request)


def test_multi_view_rejection_names_a_single_image_model(
    mocker, tmp_path, monkeypatch, synthetic_cloud
):
    """The hint used to tell people to go shoot more frames, which stopped
    being the best available advice the moment `sharp` landed."""
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.gaussian.get_model_source", return_value=object())
    mocker.patch(
        "splat.handlers.gaussian.get_reconstruction_backend",
        return_value=FakeReconstructionBackend(synthetic_cloud),
    )
    cache = get_manifest_repository()
    asset = cache.put_external(
        write_sample_png(tmp_path / "a.png", (2, 2)), kind=ManifestKind.IMAGE
    )

    with pytest.raises(SplatDomainError, match=r"--model sharp"):
        handle(GaussianRequest(inputs=[asset], model="fake-recon"))


def test_single_image_backends_get_a_hint_without_the_multi_view_advice():
    assert "viewpoints" not in _input_hint(1)
    assert "--model" not in _input_hint(1)


def test_single_image_models_are_read_from_the_catalog():
    assert _single_image_models() == ["sharp"]
