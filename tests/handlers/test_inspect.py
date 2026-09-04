from pathlib import Path

from splat.adapters.formats.ply import PlyWriter
from splat.handlers.inspect import info, validate


def test_info_reads_cloud(tmp_path: Path, synthetic_cloud):
    ply_path = tmp_path / "a.ply"
    PlyWriter().write(synthetic_cloud, ply_path)

    cloud = info(ply_path)

    assert cloud.point_count == synthetic_cloud.point_count


def test_validate_reads_cloud_and_reports_issues(tmp_path: Path, synthetic_cloud):
    ply_path = tmp_path / "a.ply"
    PlyWriter().write(synthetic_cloud, ply_path)

    result = validate(ply_path, strict=True)

    assert result.cloud.point_count == synthetic_cloud.point_count
    assert isinstance(result.issues, list)
