from pathlib import Path

from splat.adapters.formats.ply import PlyReader, PlyWriter
from splat.adapters.formats.splat_fmt import SplatFormatReader, SplatFormatWriter
from splat.application.tools.convert import ConvertUseCase


def test_convert_ply_to_splat(tmp_path: Path, synthetic_cloud):
    ply_path = tmp_path / "in.ply"
    PlyWriter().write(synthetic_cloud, ply_path)

    splat_path = tmp_path / "out.splat"
    result = ConvertUseCase(PlyReader(), SplatFormatWriter()).execute(ply_path, splat_path)

    assert splat_path.exists()
    assert result.cloud.point_count == synthetic_cloud.point_count
    assert any("SH degree" in w for w in result.warnings)


def test_convert_splat_to_ply_round_trips_through_both_formats(tmp_path: Path, synthetic_cloud):
    ply_path = tmp_path / "a.ply"
    splat_path = tmp_path / "b.splat"
    ply_path_2 = tmp_path / "c.ply"

    PlyWriter().write(synthetic_cloud, ply_path)
    ConvertUseCase(PlyReader(), SplatFormatWriter()).execute(ply_path, splat_path)
    result = ConvertUseCase(SplatFormatReader(), PlyWriter()).execute(splat_path, ply_path_2)

    assert result.warnings == []  # .ply is a lossless sink
    assert ply_path_2.exists()
