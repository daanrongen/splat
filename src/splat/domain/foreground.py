import numpy as np

from splat.domain.errors import SplatDomainError
from splat.domain.image_space import Sticker
from splat.domain.prompts import Box

BACKGROUND_AREA = 0.6  # share of the frame a mask covers to count as background
BACKGROUND_EDGE = 0.03  # share of the frame edge a mask spans to count as background
BOX_PAD = 0.05  # subject box margin, as a share of the longer image side


def canvas_mask(sticker: Sticker, height: int, width: int) -> np.ndarray:
    x0, y0, w, h = sticker.bbox
    mask = np.zeros((height, width), dtype=bool)
    mask[y0 : y0 + h, x0 : x0 + w] = sticker.rgba[:, :, 3] > 127
    return mask


def is_background(mask: np.ndarray) -> bool:
    """A mask covering most of the frame, or spanning part of its edge (a floor, a wall)."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return False
    height, width = mask.shape
    band = max(2, min(height, width) // 100)  # masks often stop a pixel short of the edge
    span_x, span_y = (xs.max() - xs.min() + 1) / width, (ys.max() - ys.min() + 1) / height
    edge = (
        span_x * ((ys.min() < band) + (ys.max() >= height - band))
        + span_y * ((xs.min() < band) + (xs.max() >= width - band))
    ) / 4
    return bool(mask.mean() > BACKGROUND_AREA or edge > BACKGROUND_EDGE)


def subject_box(stickers: list[Sticker], height: int, width: int) -> Box:
    """The box around every mask that is not background, padded a little."""
    masks = [m for s in stickers if not is_background(m := canvas_mask(s, height, width))]
    if not masks:
        raise SplatDomainError("no foreground found: every mask looks like background")
    ys, xs = np.nonzero(np.logical_or.reduce(masks))
    pad = BOX_PAD * max(height, width)
    return (
        float(max(0, xs.min() - pad)),
        float(max(0, ys.min() - pad)),
        float(min(width, xs.max() + 1 + pad)),
        float(min(height, ys.max() + 1 + pad)),
    )
