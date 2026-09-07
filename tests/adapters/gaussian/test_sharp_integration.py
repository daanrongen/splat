"""Opt-in: runs a real SHARP forward pass. ~20s and ~3GB of RAM, so it needs
both the cached checkpoint and SPLAT_INTEGRATION_TESTS=1. Partially addresses
the "tests measure the wrapper, never the call" problem in #88.
"""

import os

import numpy as np
import pytest

from splat.adapters.model_sources.huggingface import HuggingFaceModelSource
from splat.domain.value_objects import APPLE_AMLR
from tests.image_helpers import write_sample_png

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("SPLAT_INTEGRATION_TESTS") != "1",
        reason="set SPLAT_INTEGRATION_TESTS=1 to run real forward passes",
    ),
]


@pytest.fixture
def weights_path():
    source = HuggingFaceModelSource()
    path = source.local_path("apple/Sharp")
    if path is None:
        pytest.skip("apple/Sharp not cached; run `splat models pull sharp`")
    return path


def test_reconstructs_a_valid_metric_cloud_from_one_image(tmp_path, weights_path):
    from splat.adapters.gaussian.sharp import SharpBackend

    image = write_sample_png(tmp_path / "in.png", (64, 64))

    cloud = SharpBackend(weights_path=weights_path, license=APPLE_AMLR).reconstruct([image])

    # GaussianCloud's own invariants run in __post_init__, so constructing it
    # at all proves the opacity clamp worked - Apple's save_ply writes +inf
    # logits for saturated Gaussians and the domain rejects those.
    assert cloud.point_count > 0
    assert np.isfinite(cloud.opacities).all()
    assert np.isfinite(cloud.scales).all()
    assert cloud.sh_degree == 0
    assert cloud.metadata.source_model == "sharp"
    assert cloud.metadata.capture_camera_count == 1
    assert cloud.metadata.capture_camera_position == [0.0, 0.0, 0.0]
    # Metric: SHARP puts the scene in front of the camera, in metres.
    assert cloud.means[:, 2].min() > 0.0
