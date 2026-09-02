import numpy as np
import pytest

from splat.domain.errors import InvalidGaussianCloud, UnsupportedSHDegree
from splat.domain.gaussians import GaussianCloud, sh_rest_count


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
