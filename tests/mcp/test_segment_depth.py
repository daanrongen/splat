import numpy as np

from splat.domain.image_space import DepthMap, Sticker
from splat.domain.value_objects import APPLE_ASCL, MIT
from tests.image_helpers import write_sample_png


class FakeSegmentationBackend:
    name = "fake-sam"
    license = MIT

    def segment(self, image_path, *, max_stickers=None, **params) -> list[Sticker]:
        rgba = np.zeros((2, 2, 4), dtype=np.uint8)
        return [Sticker(rgba=rgba, bbox=(0, 0, 2, 2), score=0.9, area=4)]


class FakeDepthBackend:
    name = "depth-pro"
    license = APPLE_ASCL

    def estimate(self, image_path, **params) -> DepthMap:
        return DepthMap(
            depth=np.arange(6, dtype=np.float32).reshape(2, 3),
            focal_length_px=1234.5,
            field_of_view_deg=30.0,
            metadata={},
        )


class FakeCaptionBackend:
    name = "fake-captioner"
    license = MIT

    def caption(self, image_path, *, prompt: str, max_tokens: int, temperature: float):
        return f"{prompt}: small scene"


class FakeEmbeddingBackend:
    name = "fake-embedder"
    license = MIT

    def embed_image(self, image_path, **params) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)

    def embed_text(self, text: str, **params) -> np.ndarray:
        return np.array([0.0, 1.0], dtype=np.float32)


class FakeUpscaleBackend:
    name = "fake-upscaler"
    license = MIT
    supported_factors = (2, 4)

    def upscale(self, image, *, factor: int, tile: int = 0, **params):
        return np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)


def _sample_image(tmp_path) -> str:
    path = tmp_path / "scene.png"
    return str(write_sample_png(path, (3, 2)))


def test_segment_returns_one_image_per_sticker(mocker, tmp_path, monkeypatch, call_tool):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch(
        "splat.handlers.segment.get_segmentation_backend",
        return_value=FakeSegmentationBackend(),
    )

    result = call_tool("segment", image=_sample_image(tmp_path))

    assert result.is_error is False
    assert sum(1 for block in result.content if block.type == "image") == 1


def test_depth_returns_resource_content(mocker, tmp_path, monkeypatch, call_tool):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.depth.get_depth_backend", return_value=FakeDepthBackend())

    result = call_tool("depth", image=_sample_image(tmp_path))

    assert result.is_error is False
    assert any(block.type == "resource" for block in result.content)


def test_caption_returns_text_and_resource_content(mocker, tmp_path, monkeypatch, call_tool):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.caption.get_caption_backend", return_value=FakeCaptionBackend())

    result = call_tool(
        "caption", image=_sample_image(tmp_path), model="fake-captioner", prompt="Look"
    )

    assert result.is_error is False
    assert any(
        block.type == "text" and block.text == "Look: small scene" for block in result.content
    )
    assert any(block.type == "resource" for block in result.content)


def test_embed_returns_resource_content(mocker, tmp_path, monkeypatch, call_tool):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.embed.get_embedding_backend", return_value=FakeEmbeddingBackend())

    result = call_tool("embed", text="red chair", model="fake-embedder")

    assert result.is_error is False
    assert any(block.type == "text" and "text 2d float32" in block.text for block in result.content)
    assert any(block.type == "resource" for block in result.content)


def test_upscale_returns_resource_content(mocker, tmp_path, monkeypatch, call_tool):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.upscale.get_upscale_backend", return_value=FakeUpscaleBackend())

    result = call_tool("upscale", image=_sample_image(tmp_path), model="fake-upscaler", factor=2)

    assert result.is_error is False
    assert any(block.type == "resource" for block in result.content)
