from pathlib import Path

import numpy as np
import pytest

from splat.adapters.formats.ply import PlyReader, PlyWriter
from splat.domain.errors import InvalidGaussianCloud


def test_ply_roundtrip_is_lossless(tmp_path: Path, synthetic_cloud):
    out = tmp_path / "scene.ply"
    PlyWriter().write(synthetic_cloud, out)
    loaded = PlyReader().read(out)

    assert loaded.point_count == synthetic_cloud.point_count
    assert loaded.sh_degree == synthetic_cloud.sh_degree
    np.testing.assert_allclose(loaded.means, synthetic_cloud.means, atol=1e-5)
    np.testing.assert_allclose(loaded.scales, synthetic_cloud.scales, atol=1e-5)
    np.testing.assert_allclose(loaded.rotations, synthetic_cloud.rotations, atol=1e-5)
    np.testing.assert_allclose(loaded.opacities, synthetic_cloud.opacities, atol=1e-5)
    np.testing.assert_allclose(loaded.sh_dc, synthetic_cloud.sh_dc, atol=1e-5)
    np.testing.assert_allclose(loaded.sh_rest, synthetic_cloud.sh_rest, atol=1e-5)


def test_ply_writer_supports_reports_no_lossy_warnings(synthetic_cloud):
    assert PlyWriter().supports(synthetic_cloud) == []


def test_ply_roundtrip_preserves_coordinate_convention(tmp_path: Path, synthetic_cloud):
    synthetic_cloud.metadata.coordinate_convention = "colmap"
    out = tmp_path / "scene.ply"
    PlyWriter().write(synthetic_cloud, out)
    loaded = PlyReader().read(out)
    assert loaded.metadata.coordinate_convention == "colmap"
    assert loaded.metadata.up_axis == "y"


def test_ply_roundtrip_preserves_capture_camera_pose(tmp_path: Path, synthetic_cloud):
    synthetic_cloud.metadata.capture_camera_position = [1.0, 2.0, 3.0]
    synthetic_cloud.metadata.capture_camera_rotation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    synthetic_cloud.metadata.capture_camera_intrinsics = [800.0, 800.0, 320.0, 240.0, 640.0, 480.0]
    synthetic_cloud.metadata.capture_camera_count = 5
    out = tmp_path / "scene.ply"
    PlyWriter().write(synthetic_cloud, out)
    loaded = PlyReader().read(out)
    assert loaded.metadata.capture_camera_position == [1.0, 2.0, 3.0]
    assert loaded.metadata.capture_camera_rotation == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert loaded.metadata.capture_camera_intrinsics == [800.0, 800.0, 320.0, 240.0, 640.0, 480.0]
    assert loaded.metadata.capture_camera_count == 5


def test_ply_roundtrip_leaves_absent_capture_camera_pose_as_none(tmp_path: Path, synthetic_cloud):
    out = tmp_path / "scene.ply"
    PlyWriter().write(synthetic_cloud, out)
    loaded = PlyReader().read(out)
    assert loaded.metadata.capture_camera_position is None
    assert loaded.metadata.capture_camera_count is None


def test_ply_degree_zero_roundtrip(tmp_path: Path):
    from splat.domain.gaussians import GaussianCloud

    n = 3
    cloud = GaussianCloud(
        means=np.zeros((n, 3), dtype=np.float32),
        scales=np.full((n, 3), -2.0, dtype=np.float32),
        rotations=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1)),
        opacities=np.zeros(n, dtype=np.float32),
        sh_dc=np.ones((n, 3), dtype=np.float32),
        sh_rest=None,
        sh_degree=0,
    )
    out = tmp_path / "degree0.ply"
    PlyWriter().write(cloud, out)
    loaded = PlyReader().read(out)
    assert loaded.sh_degree == 0
    assert loaded.sh_rest is None


@pytest.mark.parametrize("content", [b"not a ply", b"ply\nformat ascii 1.0\nend_header\n"])
def test_reading_a_non_splat_ply_raises_a_domain_error(tmp_path: Path, content):
    path = tmp_path / "broken.ply"
    path.write_bytes(content)

    with pytest.raises(InvalidGaussianCloud, match=r"broken\.ply"):
        PlyReader().read(path)


def test_ply_roundtrip_preserves_source_cameras(tmp_path: Path, synthetic_cloud):
    cameras = [
        {"position": [1.0, 2.0, 3.0], "rotation": np.eye(3).tolist(), "intrinsics": [1.0] * 6}
    ]
    synthetic_cloud.metadata.source_cameras = cameras
    out = tmp_path / "scene.ply"
    PlyWriter().write(synthetic_cloud, out)

    assert PlyReader().read(out).metadata.source_cameras == cameras
