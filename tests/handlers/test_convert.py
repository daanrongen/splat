from pathlib import Path

from splat.adapters.formats.ply import PlyWriter
from splat.handlers.tools.convert import ConvertRequest, handle


def test_handle_converts_ply_to_splat(tmp_path: Path, synthetic_cloud):
    ply_path = tmp_path / "in.ply"
    PlyWriter().write(synthetic_cloud, ply_path)
    splat_path = tmp_path / "out.splat"

    result = handle(ConvertRequest(input_path=ply_path, output_path=splat_path))

    assert splat_path.exists()
    assert result.cloud.point_count == synthetic_cloud.point_count
