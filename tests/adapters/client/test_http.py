from pathlib import Path

import numpy as np
import pytest

from splat.adapters.formats.ply import PlyWriter
from splat.domain.asset import AssetKind
from splat.domain.errors import SplatDomainError
from splat.domain.image_space import DepthMap, Sticker
from splat.domain.value_objects import CC_BY_NC_SA_4_0, MIT
from splat.handlers.caption import CaptionRequest
from splat.handlers.depth import DepthRequest
from splat.handlers.diffuse import DiffuseRequest
from splat.handlers.gaussian import GaussianRequest
from splat.handlers.segment import SegmentRequest
from splat.handlers.tools.compress import CompressRequest
from splat.handlers.tools.convert import ConvertRequest
from splat.handlers.upscale import UpscaleRequest
from splat.image_io import read_rgb_or_rgba
from splat.registry.wiring import get_asset_cache
from tests.image_helpers import write_sample_png


class FakeDiffusionBackend:
    name = "fake-diffuser"

    def __init__(self, license) -> None:
        self.license = license

    def diffuse(self, prompt, *, output_path: Path, **params) -> Path:
        output_path.write_bytes(b"fake-png-bytes")
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
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch(
        "splat.handlers.diffuse.get_diffusion_backend",
        return_value=FakeDiffusionBackend(MIT),
    )

    result = remote_client.diffuse(DiffuseRequest(prompt="a fox", model="fake-diffuser"))

    assert result.asset.content_path.read_bytes() == b"fake-png-bytes"
    assert result.license_warning is None
    assert get_asset_cache().get(result.asset.id) is not None


def test_diffuse_surfaces_license_warning(mocker, tmp_path, monkeypatch, remote_client):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch(
        "splat.handlers.diffuse.get_diffusion_backend",
        return_value=FakeDiffusionBackend(CC_BY_NC_SA_4_0),
    )

    result = remote_client.diffuse(DiffuseRequest(prompt="a fox", model="fake-diffuser"))

    assert result.license_warning is not None


def test_diffuse_unknown_model_raises_domain_error(tmp_path, monkeypatch, remote_client):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))

    with pytest.raises(SplatDomainError, match="Unknown diffusion model"):
        remote_client.diffuse(DiffuseRequest(prompt="a fox", model="nope"))


def test_depth_stores_asset_with_parent_id(mocker, tmp_path, monkeypatch, remote_client):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.depth.get_depth_backend", return_value=FakeDepthBackend())
    cache = get_asset_cache()
    asset = cache.put_external(_sample_image(tmp_path), kind=AssetKind.IMAGE)

    results = remote_client.depth(DepthRequest(inputs=[asset]))

    assert len(results) == 1
    assert results[0].kind == AssetKind.DEPTH_MAP
    assert results[0].parent_ids == [asset.id]


def test_caption_stores_asset_with_parent_id(mocker, tmp_path, monkeypatch, remote_client):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.caption.get_caption_backend", return_value=FakeCaptionBackend())
    cache = get_asset_cache()
    asset = cache.put_external(_sample_image(tmp_path), kind=AssetKind.IMAGE)

    results = remote_client.caption(
        CaptionRequest(inputs=[asset], model="fake-captioner", prompt="Look")
    )

    assert len(results) == 1
    assert results[0].kind == AssetKind.CAPTION
    assert results[0].parent_ids == [asset.id]
    assert results[0].content_path.read_text(encoding="utf-8") == "Look: small scene"


def test_upscale_stores_asset_with_parent_id(mocker, tmp_path, monkeypatch, remote_client):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.upscale.get_upscale_backend", return_value=FakeUpscaleBackend())
    cache = get_asset_cache()
    asset = cache.put_external(_sample_image(tmp_path), kind=AssetKind.IMAGE)

    results = remote_client.upscale(
        UpscaleRequest(inputs=[asset], model="fake-upscaler", factor=2, tile=8)
    )

    assert len(results) == 1
    assert results[0].kind == AssetKind.IMAGE
    assert results[0].parent_ids == [asset.id]
    assert results[0].metadata["output_width"] == 4
    assert read_rgb_or_rgba(results[0].content_path).shape == (4, 4, 3)


def test_segment_fetches_each_sticker(mocker, tmp_path, monkeypatch, remote_client):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch(
        "splat.handlers.segment.get_segmentation_backend",
        return_value=FakeSegmentationBackend(),
    )
    cache = get_asset_cache()
    asset = cache.put_external(_sample_image(tmp_path), kind=AssetKind.IMAGE)

    results = remote_client.segment(SegmentRequest(inputs=[asset]))

    assert len(results) == 1
    assert results[0].kind == AssetKind.STICKER
    assert len(results[0].content_path.read_bytes()) > 0


def test_gaussian_writes_local_output(mocker, tmp_path, synthetic_cloud, remote_client):
    mocker.patch("splat.handlers.gaussian.get_model_source", return_value=object())
    mocker.patch(
        "splat.handlers.gaussian.get_reconstruction_backend",
        return_value=FakeReconstructionBackend(synthetic_cloud),
    )
    output_path = tmp_path / "out.ply"
    image_a, image_b = _sample_image(tmp_path), tmp_path / "b.png"
    write_sample_png(image_b, (2, 2))

    result = remote_client.gaussian(
        GaussianRequest(inputs=[image_a, image_b], output_path=output_path)
    )

    assert output_path.exists()
    assert result.cloud.point_count == synthetic_cloud.point_count


def test_tools_convert_writes_local_output(tmp_path, synthetic_cloud, remote_client):
    ply_path = tmp_path / "in.ply"
    PlyWriter().write(synthetic_cloud, ply_path)
    splat_path = tmp_path / "out.splat"

    result = remote_client.tools_convert(
        ConvertRequest(input_path=ply_path, output_path=splat_path)
    )

    assert splat_path.exists()
    assert result.cloud.point_count == synthetic_cloud.point_count


def test_tools_compress_writes_local_output(tmp_path, synthetic_cloud, remote_client):
    ply_path = tmp_path / "in.ply"
    PlyWriter().write(synthetic_cloud, ply_path)
    out_path = tmp_path / "out.ply"

    cloud = remote_client.tools_compress(CompressRequest(input_path=ply_path, output_path=out_path))

    assert out_path.exists()
    assert cloud.point_count <= synthetic_cloud.point_count


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
