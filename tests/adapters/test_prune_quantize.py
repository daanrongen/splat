from dataclasses import replace

import numpy as np
import pytest

from splat.adapters.compression.prune_quantize import PruneQuantizeCompressor


def test_web_delivery_drops_sh_rest_and_prunes(synthetic_cloud):
    result = PruneQuantizeCompressor().compress(synthetic_cloud, profile="web-delivery")
    assert result.sh_degree == 0
    assert result.sh_rest is None
    assert result.point_count <= synthetic_cloud.point_count


def test_archival_preserves_sh_and_all_points(synthetic_cloud):
    result = PruneQuantizeCompressor().compress(synthetic_cloud, profile="archival")
    assert result.sh_degree == synthetic_cloud.sh_degree
    assert result.point_count == synthetic_cloud.point_count


def test_unknown_profile_raises(synthetic_cloud):
    with pytest.raises(ValueError, match="Unknown compression profile"):
        PruneQuantizeCompressor().compress(synthetic_cloud, profile="nonsense")


def test_compress_preserves_camera_and_license_metadata(synthetic_cloud):
    cloud = replace(
        synthetic_cloud,
        metadata=replace(
            synthetic_cloud.metadata,
            coordinate_convention="colmap",
            capture_camera_position=[1.0, 2.0, 3.0],
            capture_camera_rotation=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            capture_camera_intrinsics=[1611.0, 1611.0, 256.0, 256.0, 512.0, 512.0],
            capture_camera_count=3,
        ),
    )

    result = PruneQuantizeCompressor().compress(cloud, profile="web-delivery")

    assert result.metadata.coordinate_convention == "colmap"
    assert result.metadata.capture_camera_position == [1.0, 2.0, 3.0]
    assert result.metadata.capture_camera_count == 3


def test_archival_dedupes_near_identical_points(synthetic_cloud):
    duplicated = replace(synthetic_cloud, means=synthetic_cloud.means.copy())
    duplicated.means[1] = duplicated.means[0]  # a near-identical duplicate of point 0

    result = PruneQuantizeCompressor().compress(duplicated, profile="archival")

    assert result.point_count == duplicated.point_count - 1


def test_archival_dedupe_does_not_merge_genuinely_distinct_points(synthetic_cloud):
    result = PruneQuantizeCompressor().compress(synthetic_cloud, profile="archival")
    assert result.point_count == synthetic_cloud.point_count


def test_outlier_rejection_matches_a_median_centroid(synthetic_cloud):
    # synthetic_cloud's opacities all clear web-delivery's 1/255 threshold, so
    # point_count here is driven entirely by the outlier-distance filter -
    # this pins it to the median-centroid formula normalize_gaussian_cloud
    # already uses, rather than a mean centroid that outliers can drag off.
    result = PruneQuantizeCompressor().compress(synthetic_cloud, profile="web-delivery")

    centroid = np.median(synthetic_cloud.means, axis=0)
    distances = np.linalg.norm(synthetic_cloud.means - centroid, axis=1)
    expected_count = int(np.count_nonzero(distances <= distances.mean() + 3.0 * distances.std()))

    assert result.point_count == expected_count
