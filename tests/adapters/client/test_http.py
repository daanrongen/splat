from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from splat.adapters.formats.image import read_rgb_or_rgba
from splat.adapters.formats.ply import PlyWriter
from splat.domain.errors import SplatDomainError
from splat.domain.image_space import DepthMap, Sticker
from splat.domain.manifest import ManifestKind
from splat.domain.value_objects import CC_BY_NC_SA_4_0, MIT
from splat.handlers.caption import CaptionRequest
from splat.handlers.depth import DepthRequest
from splat.handlers.diffuse import DiffuseRequest
from splat.handlers.embed import EmbedRequest
from splat.handlers.gaussian import GaussianRequest
from splat.handlers.segment import SegmentRequest
from splat.handlers.upscale import UpscaleRequest
from splat.http import gaussian as gaussian_route
from splat.registry.wiring import get_manifest_repository
from tests.image_helpers import sample_png_bytes, write_sample_png


class FakeDiffusionBackend:
    name = "fake-diffuser"

    def __init__(self, license) -> None:
        self.license = license

    def diffuse(self, prompt, *, output_path: Path, **params) -> Path:
        output_path.write_bytes(sample_png_bytes())
        return output_path


class FakeDepthBackend:
    name = "depth-pro"
    license = MIT

    def estimate(self, image_path, **params) -> DepthMap:
        return DepthMap(depth=np.ones((2, 2), dtype=np.float32), focal_length_px=10.0, metadata={})


class FakeCaptionBackend:
    name = "fake-captioner"
    license = MIT

    def caption(self, image_path: Path, *, prompt: str, max_tokens: int, temperature: float):
        return f"{prompt}: small scene"


class FakeEmbeddingBackend:
    name = "fake-embedder"
    license = MIT

    def embed_image(self, image_path: Path, **params) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)

    def embed_text(self, text: str, **params) -> np.ndarray:
        return np.array([0.0, 1.0], dtype=np.float32)


class FakeSegmentationBackend:
    name = "fake-sam"
    license = MIT

    def segment(self, image_path, *, max_stickers=None, **params) -> list[Sticker]:
        rgba = np.zeros((2, 2, 4), dtype=np.uint8)
        return [Sticker(rgba=rgba, bbox=(0, 0, 2, 2), score=0.9, area=4)]


class FakeUpscaleBackend:
    name = "fake-upscaler"
    license = MIT
    supported_factors = (2, 4)

    def upscale(self, image, *, factor: int, tile: int = 0, **params):
        return np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)


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


def _sample_image(tmp_path: Path) -> Path:
    path = tmp_path / "scene.png"
    return write_sample_png(path, (2, 2))


def test_diffuse_stores_asset_locally(mocker, tmp_path, monkeypatch, remote_client):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch(
        "splat.handlers.diffuse.get_diffusion_backend",
        return_value=FakeDiffusionBackend(MIT),
    )

    result = remote_client.diffuse(DiffuseRequest(prompt="a fox", model="fake-diffuser"))

    assert result.asset.content_path.read_bytes() == sample_png_bytes()
    assert result.license_warning is None
    assert get_manifest_repository().get(result.asset.id) is not None


def test_diffuse_surfaces_license_warning(mocker, tmp_path, monkeypatch, remote_client):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch(
        "splat.handlers.diffuse.get_diffusion_backend",
        return_value=FakeDiffusionBackend(CC_BY_NC_SA_4_0),
    )

    result = remote_client.diffuse(DiffuseRequest(prompt="a fox", model="fake-diffuser"))

    assert result.license_warning is not None


def test_diffuse_unknown_model_raises_domain_error(tmp_path, monkeypatch, remote_client):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))

    with pytest.raises(SplatDomainError, match="Unknown diffusion model"):
        remote_client.diffuse(DiffuseRequest(prompt="a fox", model="nope"))


def test_depth_stores_asset_with_parent_id(mocker, tmp_path, monkeypatch, remote_client):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.depth.get_depth_backend", return_value=FakeDepthBackend())
    cache = get_manifest_repository()
    asset = cache.put_external(_sample_image(tmp_path), kind=ManifestKind.IMAGE)

    results = remote_client.depth(DepthRequest(inputs=[asset]))

    assert len(results) == 1
    assert results[0].kind == ManifestKind.DEPTH_MAP
    assert results[0].parent_ids == [asset.id]


def test_caption_stores_asset_with_parent_id(mocker, tmp_path, monkeypatch, remote_client):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.caption.get_caption_backend", return_value=FakeCaptionBackend())
    cache = get_manifest_repository()
    asset = cache.put_external(_sample_image(tmp_path), kind=ManifestKind.IMAGE)

    results = remote_client.caption(
        CaptionRequest(inputs=[asset], model="fake-captioner", prompt="Look")
    )

    assert len(results) == 1
    assert results[0].kind == ManifestKind.CAPTION
    assert results[0].parent_ids == [asset.id]
    assert results[0].content_path.read_text(encoding="utf-8") == "Look: small scene"


def test_embed_stores_text_asset(mocker, tmp_path, monkeypatch, remote_client):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.embed.get_embedding_backend", return_value=FakeEmbeddingBackend())

    results = remote_client.embed(EmbedRequest(text="red chair", model="fake-embedder"))

    assert len(results) == 1
    assert results[0].kind == ManifestKind.EMBEDDING
    assert results[0].metadata.input_type == "text"
    assert results[0].metadata.dimension == 2
    assert results[0].metadata.text_length == len("red chair")


def test_embed_stores_image_asset_with_parent_id(mocker, tmp_path, monkeypatch, remote_client):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.embed.get_embedding_backend", return_value=FakeEmbeddingBackend())
    cache = get_manifest_repository()
    asset = cache.put_external(_sample_image(tmp_path), kind=ManifestKind.IMAGE)

    results = remote_client.embed(EmbedRequest(inputs=[asset], model="fake-embedder"))

    assert len(results) == 1
    assert results[0].kind == ManifestKind.EMBEDDING
    assert results[0].parent_ids == [asset.id]


def test_upscale_stores_asset_with_parent_id(mocker, tmp_path, monkeypatch, remote_client):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.upscale.get_upscale_backend", return_value=FakeUpscaleBackend())
    cache = get_manifest_repository()
    asset = cache.put_external(_sample_image(tmp_path), kind=ManifestKind.IMAGE)

    results = remote_client.upscale(
        UpscaleRequest(inputs=[asset], model="fake-upscaler", factor=2, tile=8)
    )

    assert len(results) == 1
    assert results[0].kind == ManifestKind.IMAGE
    assert results[0].parent_ids == [asset.id]
    assert results[0].metadata.output_width == 4
    assert read_rgb_or_rgba(results[0].content_path).shape == (4, 4, 3)


def test_segment_fetches_each_sticker(mocker, tmp_path, monkeypatch, remote_client):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch(
        "splat.handlers.segment.get_segmentation_backend",
        return_value=FakeSegmentationBackend(),
    )
    cache = get_manifest_repository()
    asset = cache.put_external(_sample_image(tmp_path), kind=ManifestKind.IMAGE)

    results = remote_client.segment(SegmentRequest(inputs=[asset]))

    assert len(results) == 1
    assert results[0].kind == ManifestKind.STICKER
    assert len(results[0].content_path.read_bytes()) > 0


def test_gaussian_stores_remote_asset(
    mocker, tmp_path, monkeypatch, synthetic_cloud, remote_client
):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.gaussian.get_model_source", return_value=object())
    mocker.patch(
        "splat.handlers.gaussian.get_reconstruction_backend",
        return_value=FakeReconstructionBackend(synthetic_cloud),
    )
    image_a, image_b = _sample_image(tmp_path), tmp_path / "b.png"
    write_sample_png(image_b, (2, 2))
    cache = get_manifest_repository()
    asset_a = cache.put_external(image_a, kind=ManifestKind.IMAGE)
    asset_b = cache.put_external(image_b, kind=ManifestKind.IMAGE)

    results = remote_client.gaussian(GaussianRequest(inputs=[asset_a, asset_b], model="fake-recon"))

    assert len(results) == 1
    assert results[0].kind == ManifestKind.GAUSSIAN_CLOUD
    assert results[0].metadata.point_count == synthetic_cloud.point_count
    assert results[0].parent_ids == [asset_a.id, asset_b.id]


def test_gaussian_forwards_orbit_frames(
    mocker, tmp_path, monkeypatch, synthetic_cloud, remote_client
):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.gaussian.get_model_source", return_value=object())
    mocker.patch(
        "splat.handlers.gaussian.get_reconstruction_backend",
        return_value=FakeReconstructionBackend(synthetic_cloud),
    )
    seen = []
    real_handle = gaussian_route.handle

    def spy(request):
        seen.append(request)
        return real_handle(replace(request, orbit_frames=None))

    mocker.patch.object(gaussian_route, "handle", side_effect=spy)
    image_a, image_b = _sample_image(tmp_path), tmp_path / "b.png"
    write_sample_png(image_b, (2, 2))
    cache = get_manifest_repository()
    inputs = [cache.put_external(p, kind=ManifestKind.IMAGE) for p in (image_a, image_b)]

    remote_client.gaussian(
        GaussianRequest(inputs=inputs, model="fake-recon", orbit_frames=6, orbit_degrees=40.0)
    )

    assert (seen[0].orbit_frames, seen[0].orbit_degrees) == (6, 40.0)


def test_info_returns_summary(tmp_path, synthetic_cloud, remote_client):
    ply_path = tmp_path / "a.ply"
    PlyWriter().write(synthetic_cloud, ply_path)

    summary = remote_client.info(ply_path)

    assert summary.points == synthetic_cloud.point_count


def test_validate_returns_summary(tmp_path, synthetic_cloud, remote_client):
    ply_path = tmp_path / "a.ply"
    PlyWriter().write(synthetic_cloud, ply_path)

    summary = remote_client.validate(ply_path, strict=True)

    assert summary.points == synthetic_cloud.point_count


def test_models_list_returns_summaries(remote_client):
    rows = remote_client.models_list()

    assert len(rows) > 0
    assert all(hasattr(r, "cached") for r in rows)


def test_models_pull_unknown_model_raises(remote_client):
    with pytest.raises(SplatDomainError):
        remote_client.models_pull("not-a-real-model")


def test_models_info_unknown_model_raises(remote_client):
    with pytest.raises(SplatDomainError):
        remote_client.models_info("not-a-real-model")


def test_models_rm_unknown_model_raises(remote_client):
    with pytest.raises(SplatDomainError):
        remote_client.models_rm("not-a-real-model")
