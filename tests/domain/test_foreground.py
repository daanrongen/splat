import numpy as np
import pytest

from splat.domain.errors import SplatDomainError
from splat.domain.foreground import foreground, is_background
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


def test_frame_and_border_masks_are_background():
    assert is_background(np.ones((SIZE, SIZE), dtype=bool))
    assert is_background(_rect(150, 200, 0, 200))
    assert not is_background(_rect(60, 140, 60, 140))


def test_foreground_merges_parts_and_drops_background_and_islands():
    body = _rect(60, 140, 60, 140)
    body[90:94, 90:94] = False  # small hole: filled
    body[70:130, 100:130] = False  # large opening: kept
    lid = _rect(40, 61, 70, 130)  # adjoins the body
    island = _rect(5, 15, 5, 15)
    stickers = [
        _sticker(m)
        for m in (np.ones((SIZE, SIZE), bool), _rect(150, 200, 0, 200), body, lid, island)
    ]

    cutout = foreground(stickers, np.zeros((SIZE, SIZE, 3), dtype=np.uint8))

    x0, y0, w, h = cutout.bbox
    assert (x0, y0, x0 + w, y0 + h) == (60, 40, 140, 140)
    alpha = cutout.rgba[:, :, 3]
    assert alpha[90 - y0 + 1, 90 - x0 + 1] > 127  # small hole filled
    assert alpha[100 - y0, 115 - x0] < 127  # large opening kept


def test_foreground_fails_when_everything_is_background():
    with pytest.raises(SplatDomainError):
        foreground([_sticker(np.ones((SIZE, SIZE), bool))], np.zeros((SIZE, SIZE, 3), np.uint8))
