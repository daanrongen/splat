import numpy as np

from splat.domain.errors import SplatDomainError
from splat.domain.image_space import Sticker

BACKGROUND_AREA = 0.6  # share of the frame a mask covers to count as background
BACKGROUND_BORDER = 0.2  # share of the frame border a mask covers to count as background
HOLE_AREA = 0.02  # holes up to this share of the subject are filled
FEATHER_SIGMA = 1.0


def canvas_mask(sticker: Sticker, height: int, width: int) -> np.ndarray:
    x0, y0, w, h = sticker.bbox
    mask = np.zeros((height, width), dtype=bool)
    mask[y0 : y0 + h, x0 : x0 + w] = sticker.rgba[:, :, 3] > 127
    return mask


def is_background(mask: np.ndarray) -> bool:
    border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    return bool(mask.mean() > BACKGROUND_AREA or border.mean() > BACKGROUND_BORDER)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    labels, count = ndimage.label(mask)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    sizes[0] = 0
    return labels == sizes.argmax()


def _fill_small_holes(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    labels, count = ndimage.label(~mask)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    outer = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    small = sizes <= HOLE_AREA * mask.sum()
    small[outer] = False
    small[0] = False
    return mask | small[labels]


def foreground(stickers: list[Sticker], image: np.ndarray) -> Sticker:
    """The main subject as one sticker: non-background masks merged and tidied."""
    import cv2

    height, width = image.shape[:2]
    masks = [m for s in stickers if not is_background(m := canvas_mask(s, height, width))]
    if not masks:
        raise SplatDomainError("no foreground found: every mask looks like background")
    merged = np.logical_or.reduce(masks)
    size = max(3, min(height, width) // 100) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    merged = cv2.morphologyEx(merged.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    merged = _fill_small_holes(_largest_component(merged))

    ys, xs = np.nonzero(merged)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    alpha = cv2.GaussianBlur(merged.astype(np.float32) * 255, (0, 0), FEATHER_SIGMA)
    rgba = np.dstack([image[:, :, :3], alpha.astype(np.uint8)])[y0:y1, x0:x1]
    return Sticker(
        rgba=rgba,
        bbox=(int(x0), int(y0), int(x1 - x0), int(y1 - y0)),
        score=float(np.mean([s.score for s in stickers])),
        area=int(merged.sum()),
    )
