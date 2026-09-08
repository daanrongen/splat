from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from splat.adapters.formats.ply import PlyWriter
from splat.adapters.formats.sog import SH_C0, SogReader, SogWriter, morton_order


def _match_by_position(a_means: np.ndarray, b_means: np.ndarray) -> np.ndarray:
    """`.sog` reorders points for spatial locality, so round-trip
    comparisons must match by nearest position rather than by index."""
    return cKDTree(a_means).query(b_means)[1]


def test_morton_order_groups_nearby_points():
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=[-10, -10, -10], scale=0.1, size=(20, 3)).astype(np.float32)
    cluster_b = rng.normal(loc=[10, 10, 10], scale=0.1, size=(20, 3)).astype(np.float32)
    points = np.concatenate([cluster_a, cluster_b])

    order = morton_order(points)
    labels = np.concatenate([np.zeros(20), np.ones(20)])[order]

    # a spatially-coherent order groups each cluster into one contiguous run
    assert (np.diff(labels) != 0).sum() <= 1


def test_sog_roundtrip_preserves_point_count_and_geometry(tmp_path: Path, synthetic_cloud):
    out = tmp_path / "scene.sog"
    writer = SogWriter()
    warnings = writer.supports(synthetic_cloud)
    assert any("SH degree" in w for w in warnings)

    writer.write(synthetic_cloud, out)
    loaded = SogReader().read(out)

    assert loaded.point_count == synthetic_cloud.point_count
    assert loaded.sh_degree == 0
    assert loaded.sh_rest is None

    match = _match_by_position(synthetic_cloud.means, loaded.means)
    np.testing.assert_allclose(loaded.means, synthetic_cloud.means[match], atol=1e-2)

    colors_in = np.clip(0.5 + SH_C0 * synthetic_cloud.sh_dc[match], 0.0, 1.0)
    colors_out = np.clip(0.5 + SH_C0 * loaded.sh_dc, 0.0, 1.0)
    np.testing.assert_allclose(colors_out, colors_in, atol=1 / 255 + 1e-3)

    np.testing.assert_allclose(
        loaded.to_activated_opacities(),
        synthetic_cloud.to_activated_opacities()[match],
        atol=1 / 255 + 1e-3,
    )


def test_sog_write_converts_colmap_coordinates_instead_of_relabelling(
    tmp_path: Path, synthetic_cloud
):
    colmap_cloud = replace(
        synthetic_cloud, metadata=replace(synthetic_cloud.metadata, coordinate_convention="colmap")
    )
    out = tmp_path / "scene.sog"
    SogWriter().write(colmap_cloud, out)
    loaded = SogReader().read(out)

    expected = colmap_cloud.means @ np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    match = _match_by_position(expected, loaded.means)
    np.testing.assert_allclose(loaded.means, expected[match], atol=1e-2)


def test_sog_is_much_smaller_than_ply_on_a_real_sized_cloud(tmp_path: Path):
    rng = np.random.default_rng(1)
    n = 5000
    from splat.domain.gaussians import GaussianCloud

    cloud = GaussianCloud(
        means=rng.normal(scale=2.0, size=(n, 3)).astype(np.float32),
        scales=rng.uniform(-4, -1, size=(n, 3)).astype(np.float32),
        rotations=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n, 1)),
        opacities=rng.uniform(-4, 4, size=n).astype(np.float32),
        sh_dc=rng.uniform(-1, 1, size=(n, 3)).astype(np.float32),
        sh_rest=rng.uniform(-0.5, 0.5, size=(n, 15, 3)).astype(np.float32),
        sh_degree=3,
    )

    ply_path = tmp_path / "scene.ply"
    sog_path = tmp_path / "scene.sog"
    PlyWriter().write(cloud, ply_path)
    SogWriter().write(cloud, sog_path)

    assert sog_path.stat().st_size < ply_path.stat().st_size / 3
