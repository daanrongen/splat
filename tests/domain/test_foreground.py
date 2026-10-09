import numpy as np
import pytest

from splat.domain.errors import SplatDomainError
from splat.domain.foreground import is_background, subject_box
from splat.domain.image_space import Sticker

SIZE = 200


def _sticker(mask: np.ndarray) -> Sticker:
    ys, xs = np.nonzero(mask)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    rgba = np.zeros((y1 - y0, x1 - x0, 4), dtype=np.uint8)
    rgba[:, :, 3] = mask[y0:y1, x0:x1] * 255
    return Sticker(rgba, (int(x0), int(y0), int(x1 - x0), int(y1 - y0)), 0.9, int(mask.sum()))


def _rect(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_frame_and_edge_masks_are_background():
    assert is_background(np.ones((SIZE, SIZE), dtype=bool))
    assert is_background(_rect(150, 200, 0, 200))
    assert is_background(_rect(150, 199, 20, 80))  # a floor patch stopping a pixel short
    assert not is_background(_rect(60, 140, 60, 140))


def test_subject_box_wraps_the_non_background_masks():
    stickers = [
        _sticker(m)
        for m in (
            np.ones((SIZE, SIZE), bool),
            _rect(150, 199, 20, 80),  # reflection
            _rect(60, 140, 60, 140),  # body
            _rect(40, 61, 70, 130),  # head
        )
    ]

    x0, y0, x1, y1 = subject_box(stickers, SIZE, SIZE)

    assert (x0, y0, x1, y1) == (50.0, 30.0, 150.0, 150.0)


def test_subject_box_fails_when_everything_is_background():
    with pytest.raises(SplatDomainError):
        subject_box([_sticker(np.ones((SIZE, SIZE), bool))], SIZE, SIZE)
