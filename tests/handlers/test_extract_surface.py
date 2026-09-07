from pathlib import Path

from splat.adapters.formats.ply import PlyWriter
from splat.handlers.tools.extract_surface import ExtractSurfaceRequest, handle
from tests.adapters.mesh.test_poisson import _sphere_cloud


def test_handle_extracts_and_writes(tmp_path: Path):
    cloud = _sphere_cloud()
    ply_path = tmp_path / "in.ply"
    PlyWriter().write(cloud, ply_path)
    out_path = tmp_path / "out.obj"

    result = handle(ExtractSurfaceRequest(input_path=ply_path, output_path=out_path, depth=6))

    assert out_path.exists()
    assert result.input_point_count == cloud.point_count
    assert result.vertex_count > 0
    assert result.face_count > 0


def test_handle_infers_format_from_output_suffix(tmp_path: Path):
    cloud = _sphere_cloud()
    ply_path = tmp_path / "in.ply"
    PlyWriter().write(cloud, ply_path)
    out_path = tmp_path / "out.glb"

    handle(ExtractSurfaceRequest(input_path=ply_path, output_path=out_path, depth=6))

    assert out_path.exists()
