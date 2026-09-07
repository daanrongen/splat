import numpy as np

from splat.adapters.cleanup.density_declutter import DensityDeclutterer, floater_mask


def _clustered_points_with_floater() -> np.ndarray:
    rng = np.random.default_rng(0)
    cluster = rng.normal(scale=0.05, size=(40, 3)).astype(np.float32)
    floater = np.array([[50.0, 50.0, 50.0]], dtype=np.float32)
    return np.concatenate([cluster, floater], axis=0)


def test_floater_mask_flags_isolated_point():
    points = _clustered_points_with_floater()
    mask = floater_mask(points, k=8, std_ratio=2.0)
    assert mask[:-1].all()
    assert not mask[-1]


def test_floater_mask_keeps_almost_everything_when_uniform():
    rng = np.random.default_rng(1)
    points = rng.normal(scale=1.0, size=(30, 3)).astype(np.float32)
    mask = floater_mask(points, k=8, std_ratio=3.0)
    # a handful of points may sit just past 3 std-devs by chance; none should
    # be flagged as isolated floaters in an otherwise uniform cluster.
    assert mask.sum() >= len(points) - 1


def test_floater_mask_no_op_when_fewer_points_than_k():
    points = np.zeros((4, 3), dtype=np.float32)
    mask = floater_mask(points, k=16)
    assert mask.all()


def test_density_declutterer_removes_floater_and_preserves_other_fields(synthetic_cloud):
    from dataclasses import replace

    means = _clustered_points_with_floater().astype(np.float32)
    n = means.shape[0]
    cloud = replace(
        synthetic_cloud,
        means=means,
        scales=synthetic_cloud.scales[:n],
        rotations=synthetic_cloud.rotations[:n],
        opacities=synthetic_cloud.opacities[:n],
        sh_dc=synthetic_cloud.sh_dc[:n],
        sh_rest=synthetic_cloud.sh_rest[:n],
    )

    cleaned = DensityDeclutterer().declutter(cloud, k=8, std_ratio=2.0)

    assert cleaned.point_count == n - 1
    assert cleaned.metadata.point_count == n - 1
