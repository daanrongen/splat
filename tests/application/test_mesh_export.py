from io import BytesIO

import numpy as np
import trimesh

from splat.application.pipeline import shape_to_mesh_bytes
from splat.domain.image_space import Shape3D


def _triangle(color: int) -> Shape3D:
    return Shape3D(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        colors=np.full((3, 3), color, dtype=np.uint8),
    )


def _vertex_color(data: bytes, file_type: str) -> int:
    mesh = trimesh.load(BytesIO(data), file_type=file_type, force="mesh", process=False)
    return int(mesh.visual.vertex_colors[0][0])


def test_glb_vertex_colors_are_stored_linear():
    assert _vertex_color(shape_to_mesh_bytes(_triangle(128), "glb"), "glb") == 55


def test_ply_vertex_colors_keep_their_srgb_values():
    assert _vertex_color(shape_to_mesh_bytes(_triangle(128), "ply"), "ply") == 128
