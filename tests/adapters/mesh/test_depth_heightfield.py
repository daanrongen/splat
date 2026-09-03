from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from splat.adapters.mesh.depth_heightfield import DepthHeightfieldBackend
from splat.domain.errors import SplatDomainError
from splat.domain.image_space import DepthMap
from splat.domain.value_objects import MIT


def _sample_image(tmp_path: Path, size=(4, 4)) -> Path:
    path = tmp_path / "scene.png"
    Image.new("RGB", size).save(path)
    return path


def test_predict_requires_depth_map(tmp_path):
    backend = DepthHeightfieldBackend(license=MIT)

    with pytest.raises(SplatDomainError, match="requires a depth map"):
        backend.predict(_sample_image(tmp_path))


def test_predict_requires_focal_length(tmp_path):
    backend = DepthHeightfieldBackend(license=MIT)
    depth_map = DepthMap(depth=np.ones((4, 4), dtype=np.float32), focal_length_px=None)

    with pytest.raises(SplatDomainError, match="focal length"):
        backend.predict(_sample_image(tmp_path), depth_map=depth_map)


def test_predict_builds_textured_mesh(tmp_path):
    backend = DepthHeightfieldBackend(license=MIT)
    depth = np.full((4, 4), 2.0, dtype=np.float32)
    depth_map = DepthMap(depth=depth, focal_length_px=50.0, field_of_view_deg=30.0)

    shape = backend.predict(_sample_image(tmp_path), depth_map=depth_map)

    assert shape.vertices.shape == (16, 3)
    assert shape.faces.shape[1] == 3
    assert len(shape.faces) > 0
    assert shape.uv.shape == (16, 2)
    assert shape.texture.shape == (4, 4, 3)
    assert shape.metadata["vertex_count"] == 16


def test_predict_drops_faces_at_depth_discontinuities(tmp_path):
    backend = DepthHeightfieldBackend(license=MIT)
    depth = np.full((4, 4), 2.0, dtype=np.float32)
    depth[:, 2:] = 20.0  # a hard step down the middle of the grid
    depth_map = DepthMap(depth=depth, focal_length_px=50.0)

    shape = backend.predict(_sample_image(tmp_path), depth_map=depth_map)

    full_grid_faces = 2 * (4 - 1) * (4 - 1)
    assert len(shape.faces) < full_grid_faces
