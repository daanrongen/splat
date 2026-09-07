from pathlib import Path

from splat.adapters.formats.ply import PlyWriter
from splat.handlers.tools.declutter import DeclutterRequest, handle


def test_handle_declutters_and_writes(tmp_path: Path, synthetic_cloud):
    ply_path = tmp_path / "in.ply"
    PlyWriter().write(synthetic_cloud, ply_path)
    out_path = tmp_path / "out.ply"

    cloud = handle(DeclutterRequest(input_path=ply_path, output_path=out_path))

    assert out_path.exists()
    assert cloud.point_count <= synthetic_cloud.point_count
