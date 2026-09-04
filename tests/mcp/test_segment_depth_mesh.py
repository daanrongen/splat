import numpy as np
from PIL import Image

from splat.domain.image_space import DepthMap, Shape3D, Sticker
from splat.domain.value_objects import APPLE_ASCL, MIT


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


class FakeMeshBackend:
    name = "triposr"
    license = MIT

    def predict(self, image_path, **params) -> Shape3D:
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
        return Shape3D(vertices=vertices, faces=faces, metadata={"face_count": 2})


def _sample_image(tmp_path) -> str:
    path = tmp_path / "scene.png"
    Image.new("RGB", (3, 2)).save(path)
    return str(path)


def test_segment_returns_one_image_per_sticker(mocker, tmp_path, monkeypatch, call_tool):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch(
        "splat.handlers.segment.get_segmentation_backend",
        return_value=FakeSegmentationBackend(),
    )

    result = call_tool("segment", image=_sample_image(tmp_path))

    assert result.is_error is False
    assert sum(1 for block in result.content if block.type == "image") == 1


def test_depth_returns_resource_content(mocker, tmp_path, monkeypatch, call_tool):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.depth.get_depth_backend", return_value=FakeDepthBackend())

    result = call_tool("depth", image=_sample_image(tmp_path))

    assert result.is_error is False
    assert any(block.type == "resource" for block in result.content)


def test_mesh_returns_resource_content(mocker, tmp_path, monkeypatch, call_tool):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.mesh.get_mesh_backend", return_value=FakeMeshBackend())

    result = call_tool("mesh", image=_sample_image(tmp_path))

    assert result.is_error is False
    assert any(block.type == "resource" for block in result.content)
