import numpy as np
import pytest

from splat.domain.errors import InvalidGaussianCloud, UnsupportedSHDegree
from splat.domain.gaussians import (
    GaussianCloud,
    normalize_gaussian_cloud,
    sh_rest_count,
    to_convention,
    with_view,
)


def make_cloud(n: int = 4, sh_degree: int = 0, **overrides) -> GaussianCloud:
    kwargs = {
        "means": np.zeros((n, 3), dtype=np.float32),
        "scales": np.zeros((n, 3), dtype=np.float32),
        "rotations": np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1)),
        "opacities": np.zeros(n, dtype=np.float32),
        "sh_dc": np.zeros((n, 3), dtype=np.float32),
        "sh_degree": sh_degree,
    }
    if sh_degree > 0:
        kwargs["sh_rest"] = np.zeros((n, sh_rest_count(sh_degree), 3), dtype=np.float32)
    kwargs.update(overrides)
    return GaussianCloud(**kwargs)


def test_valid_cloud_degree_0():
    cloud = make_cloud(n=5)
    assert cloud.point_count == 5
    assert cloud.metadata.point_count == 5


def test_valid_cloud_degree_2():
    cloud = make_cloud(n=3, sh_degree=2)
    assert cloud.sh_rest.shape == (3, 8, 3)


def test_mismatched_leading_dimension_raises():
    with pytest.raises(InvalidGaussianCloud):
        make_cloud(n=4, scales=np.zeros((3, 3), dtype=np.float32))


def test_sh_rest_required_for_nonzero_degree():
    with pytest.raises(InvalidGaussianCloud):
        make_cloud(n=4, sh_degree=1, sh_rest=None)


def test_sh_rest_forbidden_for_degree_zero():
    with pytest.raises(InvalidGaussianCloud):
        make_cloud(n=4, sh_degree=0, sh_rest=np.zeros((4, 3, 3), dtype=np.float32))


def test_invalid_sh_degree_raises():
    with pytest.raises(UnsupportedSHDegree):
        make_cloud(n=4, sh_degree=4)


def test_nan_scales_rejected():
    with pytest.raises(InvalidGaussianCloud):
        make_cloud(n=4, scales=np.full((4, 3), np.nan, dtype=np.float32))


def test_to_linear_scales_log_activation():
    cloud = make_cloud(n=2, scales=np.zeros((2, 3), dtype=np.float32))
    np.testing.assert_allclose(cloud.to_linear_scales(), np.ones((2, 3), dtype=np.float32))


def test_to_linear_scales_linear_activation_passthrough():
    scales = np.full((2, 3), 2.0, dtype=np.float32)
    cloud = make_cloud(n=2, scales=scales, scale_activation="linear")
    np.testing.assert_allclose(cloud.to_linear_scales(), scales)


def test_to_activated_opacities_logit():
    cloud = make_cloud(n=2, opacities=np.zeros(2, dtype=np.float32))
    np.testing.assert_allclose(cloud.to_activated_opacities(), np.full(2, 0.5, dtype=np.float32))


def test_to_activated_opacities_linear_passthrough():
    opacities = np.full(2, 0.7, dtype=np.float32)
    cloud = make_cloud(n=2, opacities=opacities, opacity_activation="linear")
    np.testing.assert_allclose(cloud.to_activated_opacities(), opacities)


def _offset_cloud(center: np.ndarray, radius: float) -> GaussianCloud:
    offsets = np.array(
        [[radius, 0, 0], [-radius, 0, 0], [0, radius, 0], [0, -radius, 0]], dtype=np.float32
    )
    return make_cloud(n=4, means=(center + offsets).astype(np.float32))


def test_normalize_recenters_on_median():
    cloud = _offset_cloud(center=np.array([10.0, -5.0, 2.0]), radius=2.0)
    normalized = normalize_gaussian_cloud(cloud)
    np.testing.assert_allclose(np.median(normalized.means, axis=0), np.zeros(3), atol=1e-5)


def test_normalize_rescales_to_target_radius():
    cloud = _offset_cloud(center=np.array([0.0, 0.0, 0.0]), radius=2.0)
    normalized = normalize_gaussian_cloud(cloud, target_radius=1.0)
    radii = np.linalg.norm(normalized.means, axis=1)
    np.testing.assert_allclose(np.median(radii), 1.0, atol=1e-5)


def test_normalize_applies_same_scale_factor_to_gaussian_scales():
    cloud = _offset_cloud(center=np.array([0.0, 0.0, 0.0]), radius=2.0)
    normalized = normalize_gaussian_cloud(cloud, target_radius=1.0)
    # radius 2.0 -> target 1.0 means a 0.5 scale factor; input scales are all
    # zero (log-scale => linear 1.0), so linear scale should now be 0.5.
    np.testing.assert_allclose(normalized.to_linear_scales(), np.full((4, 3), 0.5), rtol=1e-5)


def test_normalize_preserves_rotations_opacities_and_sh():
    cloud = _offset_cloud(center=np.array([1.0, 2.0, 3.0]), radius=2.0)
    normalized = normalize_gaussian_cloud(cloud)
    np.testing.assert_array_equal(normalized.rotations, cloud.rotations)
    np.testing.assert_array_equal(normalized.opacities, cloud.opacities)
    np.testing.assert_array_equal(normalized.sh_dc, cloud.sh_dc)


def test_normalize_transforms_capture_camera_position():
    from splat.domain.gaussians import GaussianCloudMetadata

    cloud = _offset_cloud(center=np.array([10.0, -5.0, 2.0]), radius=2.0)
    cloud.metadata = GaussianCloudMetadata(capture_camera_position=[11.0, -5.0, 2.0])
    normalized = normalize_gaussian_cloud(cloud, target_radius=1.0)
    # centroid [10,-5,2], scale factor 0.5 (radius 2.0 -> target 1.0):
    # (11,-5,2) - (10,-5,2) = (1,0,0); * 0.5 = (0.5, 0, 0)
    np.testing.assert_allclose(
        normalized.metadata.capture_camera_position, [0.5, 0.0, 0.0], atol=1e-5
    )


def test_normalize_leaves_absent_capture_camera_pose_as_none():
    cloud = _offset_cloud(center=np.array([1.0, 2.0, 3.0]), radius=2.0)
    normalized = normalize_gaussian_cloud(cloud)
    assert normalized.metadata.capture_camera_position is None


def test_normalize_degenerate_zero_radius_is_a_no_op_scale():
    means = np.full((4, 3), 7.0, dtype=np.float32)
    cloud = make_cloud(n=4, means=means)
    normalized = normalize_gaussian_cloud(cloud)
    np.testing.assert_allclose(normalized.means, np.zeros((4, 3)), atol=1e-6)
    np.testing.assert_allclose(normalized.to_linear_scales(), cloud.to_linear_scales())


def test_to_convention_is_a_noop_for_the_same_convention(synthetic_cloud):
    synthetic_cloud.metadata.coordinate_convention = "colmap"

    assert to_convention(synthetic_cloud, "colmap") is synthetic_cloud


def test_to_convention_flips_positions_and_marks_y_up(synthetic_cloud):
    synthetic_cloud.metadata.coordinate_convention = "colmap"
    before = synthetic_cloud.means.copy()

    converted = to_convention(synthetic_cloud, "opengl")

    np.testing.assert_allclose(converted.means, before * [1.0, -1.0, -1.0])
    assert converted.metadata.coordinate_convention == "opengl"
    assert converted.metadata.up_axis == "y"


def test_to_convention_round_trips(synthetic_cloud):
    synthetic_cloud.metadata.coordinate_convention = "colmap"

    back = to_convention(to_convention(synthetic_cloud, "opengl"), "colmap")

    np.testing.assert_allclose(back.means, synthetic_cloud.means)
    np.testing.assert_allclose(back.rotations, synthetic_cloud.rotations)
    assert back.metadata.coordinate_convention == "colmap"


def test_to_convention_quaternion_flip_matches_a_matrix_round_trip(synthetic_cloud):
    """The quaternion shortcut (negate the flipped axes' components) has to
    agree with conjugating the rotation matrix, or per-Gaussian orientation
    silently diverges from position."""
    from scipy.spatial.transform import Rotation

    synthetic_cloud.metadata.coordinate_convention = "colmap"
    flip = np.diag([1.0, -1.0, -1.0])

    converted = to_convention(synthetic_cloud, "opengl")

    for original, flipped in zip(synthetic_cloud.rotations, converted.rotations, strict=True):
        # GaussianCloud stores (w, x, y, z); scipy wants (x, y, z, w).
        expected = flip @ Rotation.from_quat(np.roll(original, -1)).as_matrix() @ flip
        actual = Rotation.from_quat(np.roll(flipped, -1)).as_matrix()
        np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_to_convention_moves_the_camera_pose_with_the_cloud(synthetic_cloud):
    """A pose left in the old frame is exactly how the camera ends up aimed
    away from the scene."""
    synthetic_cloud.metadata.coordinate_convention = "colmap"
    synthetic_cloud.metadata.capture_camera_position = [1.0, 2.0, 3.0]
    synthetic_cloud.metadata.capture_camera_rotation = np.eye(3).tolist()

    converted = to_convention(synthetic_cloud, "opengl")

    assert converted.metadata.capture_camera_position == [1.0, -2.0, -3.0]
    np.testing.assert_allclose(converted.metadata.capture_camera_rotation, np.eye(3))


def _camera(position):
    return {"position": position, "rotation": np.eye(3).tolist(), "intrinsics": [1, 1, 0, 0, 2, 2]}


def test_to_convention_moves_every_source_camera(synthetic_cloud):
    synthetic_cloud.metadata.coordinate_convention = "colmap"
    synthetic_cloud.metadata.source_cameras = [_camera([1.0, 2.0, 3.0]), _camera([0.0, 1.0, 0.0])]

    converted = to_convention(synthetic_cloud, "opengl")

    positions = [c["position"] for c in converted.metadata.source_cameras]
    assert positions == [[1.0, -2.0, -3.0], [0.0, -1.0, 0.0]]


def test_with_view_makes_a_source_camera_the_capture_pose(synthetic_cloud):
    synthetic_cloud.metadata.source_cameras = [_camera([1.0, 0.0, 0.0]), _camera([0.0, 5.0, 0.0])]

    viewed = with_view(synthetic_cloud, 1)

    assert viewed.metadata.capture_camera_position == [0.0, 5.0, 0.0]
    with pytest.raises(InvalidGaussianCloud, match="2 source camera"):
        with_view(synthetic_cloud, 2)
