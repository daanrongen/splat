"""Real forward pass through each cataloged segmentation backend. Opt in with
SPLAT_INTEGRATION_TESTS=1; needs the weights (`splat models pull <model>`).
"""

import os
from pathlib import Path

import pytest

from splat.registry.wiring import get_segmentation_backend

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("SPLAT_INTEGRATION_TESTS") != "1",
        reason="set SPLAT_INTEGRATION_TESTS=1 to run a real segmentation forward pass",
    ),
]

_IMAGE = Path(__file__).parents[3] / "docs" / "images" / "01-diffuse-sdxl-turbo.png"


@pytest.mark.parametrize("model", ["sam-mlx", "sam2-coreml"])
def test_segment_produces_stickers(model):
    stickers = get_segmentation_backend(model).segment(_IMAGE, max_stickers=3)

    assert 1 <= len(stickers) <= 3
    for sticker in stickers:
        x, y, w, h = sticker.bbox
        assert sticker.rgba.shape == (h, w, 4)
        assert x >= 0 and y >= 0 and x + w <= 512 and y + h <= 512
        assert sticker.area > 0
