import numpy as np
import pytest

from splat.application.tools import normalize_color


def test_execute_matches_channel_means_to_cohort_median():
    dark = np.full((4, 4, 3), 50, dtype=np.uint8)
    bright = np.full((4, 4, 3), 150, dtype=np.uint8)
    mid = np.full((4, 4, 3), 100, dtype=np.uint8)

    corrected = normalize_color.execute([dark, bright, mid])

    means = [image.reshape(-1, 3).mean() for image in corrected]
    # cohort median is 100 - the mid image should land ~unchanged, and dark/bright
    # should both be pulled toward it.
    assert means[2] == pytest.approx(100, abs=1)
    assert means[0] > 50
    assert means[1] < 150


def test_execute_preserves_alpha_channel_unchanged():
    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    rgba[..., :3] = 200
    rgba[..., 3] = 77

    corrected = normalize_color.execute([rgba])[0]

    assert corrected.shape == (2, 2, 4)
    np.testing.assert_array_equal(corrected[..., 3], np.full((2, 2), 77))


def test_execute_is_no_op_for_a_single_uniform_image():
    image = np.full((3, 3, 3), 120, dtype=np.uint8)

    corrected = normalize_color.execute([image])[0]

    np.testing.assert_array_equal(corrected, image)


def test_execute_handles_near_black_image_without_dividing_by_zero():
    black = np.zeros((2, 2, 3), dtype=np.uint8)
    bright = np.full((2, 2, 3), 200, dtype=np.uint8)

    corrected = normalize_color.execute([black, bright])

    assert np.all(np.isfinite(corrected[0]))
