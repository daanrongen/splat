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
from tests.image_helpers import sample_rgb, write_sample_png


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


class FakeRenderBackend:
    name = "fake-render"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def render(self, cloud, output_path, *, on_progress=None, **params):
        self.calls.append(params)
        from splat.adapters.formats.image import write_png

        write_png(output_path, sample_rgb((2, 2)))


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


def test_orbit_frames_renders_a_symmetric_azimuth_sweep(
    mocker, tmp_path, monkeypatch, synthetic_cloud
):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.gaussian.get_model_source", return_value=object())
    mocker.patch(
        "splat.handlers.gaussian.get_reconstruction_backend",
        return_value=FakeReconstructionBackend(synthetic_cloud),
    )
    render_backend = FakeRenderBackend()
    mocker.patch("splat.handlers.gaussian.get_render_backend", return_value=render_backend)
    cache = get_manifest_repository()
    a = cache.put_external(write_sample_png(tmp_path / "a.png", (2, 2)), kind=ManifestKind.IMAGE)
    b = cache.put_external(write_sample_png(tmp_path / "b.png", (2, 2)), kind=ManifestKind.IMAGE)
    request = GaussianRequest(inputs=[a, b], model="fake-recon", orbit_frames=3, orbit_degrees=30.0)

    results = handle(request)

    assert len(results) == 4
    cloud, frames = results[0], results[1:]
    assert cloud.kind == ManifestKind.GAUSSIAN_CLOUD
    assert [frame.kind for frame in frames] == [ManifestKind.IMAGE] * 3
    assert all(frame.parent_ids == [cloud.id] for frame in frames)
    assert [call["azimuth"] for call in render_backend.calls] == [-15.0, 0.0, 15.0]


def test_orbit_frames_rejects_a_sweep_of_one(mocker, tmp_path, monkeypatch, synthetic_cloud):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.gaussian.get_model_source", return_value=object())
    mocker.patch(
        "splat.handlers.gaussian.get_reconstruction_backend",
        return_value=FakeReconstructionBackend(synthetic_cloud),
    )
    cache = get_manifest_repository()
    a = cache.put_external(write_sample_png(tmp_path / "a.png", (2, 2)), kind=ManifestKind.IMAGE)
    b = cache.put_external(write_sample_png(tmp_path / "b.png", (2, 2)), kind=ManifestKind.IMAGE)
    request = GaussianRequest(inputs=[a, b], model="fake-recon", orbit_frames=1)

    with pytest.raises(SplatDomainError, match="at least 2"):
        handle(request)


def test_declutter_drops_an_isolated_floater(mocker, tmp_path, monkeypatch, synthetic_cloud):
    from dataclasses import replace

    import numpy as np

    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    means = synthetic_cloud.means.copy()
    means[0] = [1000.0, 1000.0, 1000.0]
    cloud = replace(synthetic_cloud, means=means.astype(np.float32))
    mocker.patch("splat.handlers.gaussian.get_model_source", return_value=object())
    mocker.patch(
        "splat.handlers.gaussian.get_reconstruction_backend",
        return_value=FakeReconstructionBackend(cloud),
    )
    cache = get_manifest_repository()
    images = [
        cache.put_external(write_sample_png(tmp_path / f"{n}.png", (2, 2)), kind=ManifestKind.IMAGE)
        for n in "ab"
    ]

    kept = handle(GaussianRequest(inputs=images, model="fake-recon"))[0]
    cleaned = handle(GaussianRequest(inputs=images, model="fake-recon", declutter=True))[0]

    assert cleaned.metadata.point_count == kept.metadata.point_count - 1
    assert cleaned.params["declutter"] is True


def test_sharp_cache_key_ignores_device_and_other_models_flags(
    mocker, tmp_path, monkeypatch, synthetic_cloud
):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.gaussian.get_model_source", return_value=object())
    backend = FakeReconstructionBackend(synthetic_cloud)
    mocker.patch("splat.handlers.gaussian.get_reconstruction_backend", return_value=backend)
    image = get_manifest_repository().put_external(
        write_sample_png(tmp_path / "a.png", (2, 2)), kind=ManifestKind.IMAGE
    )

    first = handle(GaussianRequest(inputs=[image], model="sharp", device="cpu"))[0]
    second = handle(
        GaussianRequest(inputs=[image], model="sharp", device="mps", quality="high", seed=7)
    )[0]

    assert first.id == second.id
    assert set(first.params) == {"device", "focal_35mm", "model"}


def test_declutter_survives_the_per_model_param_filter(
    mocker, tmp_path, monkeypatch, synthetic_cloud
):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.gaussian.get_model_source", return_value=object())
    mocker.patch(
        "splat.handlers.gaussian.get_reconstruction_backend",
        return_value=FakeReconstructionBackend(synthetic_cloud),
    )
    image = get_manifest_repository().put_external(
        write_sample_png(tmp_path / "a.png", (2, 2)), kind=ManifestKind.IMAGE
    )

    result = handle(GaussianRequest(inputs=[image], model="sharp", declutter=True))[0]

    assert result.params["declutter"] is True
    assert "normalize_color" not in result.params


def test_score_records_psnr_against_the_cameras_own_input(
    mocker, tmp_path, monkeypatch, synthetic_cloud
):
    from splat.adapters.formats.image import write_png

    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    synthetic_cloud.metadata.source_cameras = [
        {"position": [0, 0, 0], "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
         "intrinsics": [2, 2, 1, 1, 2, 2], "input": 1}
    ]  # fmt: skip
    mocker.patch("splat.handlers.gaussian.get_model_source", return_value=object())
    mocker.patch(
        "splat.handlers.gaussian.get_reconstruction_backend",
        return_value=FakeReconstructionBackend(synthetic_cloud),
    )
    render_backend = FakeRenderBackend()
    mocker.patch("splat.handlers.gaussian.get_render_backend", return_value=render_backend)
    cache = get_manifest_repository()
    black = cache.put_external(
        write_sample_png(tmp_path / "a.png", (2, 2)), kind=ManifestKind.IMAGE
    )
    white_path = tmp_path / "b.png"
    write_png(white_path, sample_rgb((2, 2), (255, 255, 255)))
    white = cache.put_external(white_path, kind=ManifestKind.IMAGE)

    cloud = handle(GaussianRequest(inputs=[black, white], model="fake-recon", score=True))[0]

    # The fake renders black, so only the white input (camera 1's) scores 0 dB.
    assert cloud.metadata.quality["psnr"] == 0.0
    assert "needle_ratio" in cloud.metadata.quality
    assert render_backend.calls[0]["width"] == 2
    assert cache.get(cloud.id).metadata.quality["psnr"] == 0.0
    assert cache.children(cloud.id)[0].kind == ManifestKind.IMAGE
