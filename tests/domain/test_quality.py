import numpy as np

from splat.domain.gaussians import GaussianCloud, GaussianCloudMetadata
from splat.domain.quality import cloud_stats, image_scores


def _cloud(means, scales, opacities, cameras=None) -> GaussianCloud:
    n = len(means)
    return GaussianCloud(
        means=np.asarray(means, dtype=np.float32),
        scales=np.log(np.asarray(scales, dtype=np.float32)),
        rotations=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1)),
        opacities=np.asarray(opacities, dtype=np.float32),
        sh_dc=np.zeros((n, 3), dtype=np.float32),
        sh_degree=0,
        opacity_activation="linear",
        metadata=GaussianCloudMetadata(coordinate_convention="colmap", source_cameras=cameras),
    )


CAMERA = {
    "position": [0, 0, 0],
    "rotation": np.eye(3).tolist(),
    "intrinsics": [10, 10, 5, 5, 10, 10],
}


def test_cloud_stats_flags_transparent_needles_and_unseen_gaussians():
    cloud = _cloud(
        means=[[0, 0, 5], [0, 0, 5], [0, 0, -5], [100, 0, 5]],
        scales=[[1, 1, 1], [1, 1, 0.01], [1, 1, 1], [1, 1, 1]],
        opacities=[0.9, 0.9, 0.01, 0.9],
        cameras=[CAMERA],
    )

    stats = cloud_stats(cloud)

    assert stats["transparent_ratio"] == 0.25
    assert stats["needle_ratio"] == 0.25
    assert stats["out_of_view_ratio"] == 0.5  # one behind the camera, one beside it
    assert stats["opacity_histogram"] == [0.25, 0.0, 0.0, 0.0, 0.75]


def test_out_of_view_is_unknown_without_cameras():
    assert cloud_stats(_cloud([[0, 0, 1]], [[1, 1, 1]], [0.5]))["out_of_view_ratio"] is None


def test_image_scores_match_scikit_image():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    noisy = np.clip(image * 0.7 + rng.normal(0, 20, image.shape) + 30, 0, 255).astype(np.uint8)

    assert image_scores(image, image) == {"psnr": float("inf"), "ssim": 1.0}
    # skimage: peak_signal_noise_ratio, and structural_similarity without sample covariance
    assert image_scores(noisy, image) == {"psnr": 18.14, "ssim": 0.8946}
