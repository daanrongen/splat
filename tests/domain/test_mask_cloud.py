import numpy as np
import pytest

from splat.domain.errors import InvalidGaussianCloud
from splat.domain.gaussians import GaussianCloud, mask_cloud

CAMERA = {
    "position": [0.0, 0.0, 0.0],
    "rotation": np.eye(3).tolist(),
    "intrinsics": [100.0, 100.0, 50.0, 50.0, 100.0, 100.0],
    "input": 0,
}


def _cloud(means: list[list[float]], cameras: list[dict] | None = None) -> GaussianCloud:
    n = len(means)
    cloud = GaussianCloud(
        means=np.array(means, dtype=np.float32),
        scales=np.zeros((n, 3), dtype=np.float32),
        rotations=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1)),
        opacities=np.zeros(n, dtype=np.float32),
        sh_dc=np.zeros((n, 3), dtype=np.float32),
    )
    cloud.metadata.coordinate_convention = "colmap"
    cloud.metadata.source_cameras = cameras
    return cloud


def test_mask_keeps_only_gaussians_projecting_inside_it():
    mask = np.zeros((100, 100), dtype=bool)
    mask[:, :50] = True  # left half of the frame
    cloud = _cloud(
        [
            [-0.25, 0.0, 1.0],  # u = 25: inside
            [0.25, 0.0, 1.0],  # u = 75: outside the mask
            [-0.25, 0.0, -1.0],  # behind the camera
            [-5.0, 0.0, 1.0],  # outside the frame
        ],
        [CAMERA],
    )

    masked = mask_cloud(cloud, mask)

    assert masked.means.tolist() == [[-0.25, 0.0, 1.0]]
    assert masked.point_count == 1


def test_mask_scales_to_the_mask_resolution():
    mask = np.zeros((50, 50), dtype=bool)  # half the camera's 100 px
    mask[:, :25] = True
    cloud = _cloud([[-0.25, 0.0, 1.0], [0.25, 0.0, 1.0]], [CAMERA])

    assert mask_cloud(cloud, mask).point_count == 1


def test_mask_needs_a_source_camera():
    with pytest.raises(InvalidGaussianCloud):
        mask_cloud(_cloud([[0.0, 0.0, 1.0]]), np.ones((10, 10), dtype=bool))
