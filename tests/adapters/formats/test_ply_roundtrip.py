from pathlib import Path

import numpy as np

from splat.adapters.formats.ply import PlyReader, PlyWriter


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
