from io import BytesIO
from pathlib import Path

import numpy as np

from splat.adapters.formats.image import encode_png, write_png


def sample_rgb(
    size: tuple[int, int] = (2, 2),
    color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    width, height = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = color
    return image


def sample_rgba(
    size: tuple[int, int] = (2, 2),
    color: tuple[int, int, int, int] = (0, 0, 0, 255),
) -> np.ndarray:
    width, height = size
    image = np.zeros((height, width, 4), dtype=np.uint8)
    image[:, :] = color
    return image


def write_sample_png(path: Path, size: tuple[int, int] = (2, 2)) -> Path:
    write_png(path, sample_rgb(size))
    return path


def sample_png_bytes(size: tuple[int, int] = (2, 2)) -> bytes:
    return encode_png(sample_rgb(size))


def tiny_png(tag: str) -> bytes:
    """A valid 1x1 RGB PNG whose bytes differ per tag."""
    return encode_png(sample_rgb((1, 1), tuple(tag.encode().ljust(3, b"\0")[:3])))


def depth_npy(size: tuple[int, int] = (2, 2)) -> bytes:
    buf = BytesIO()
    np.save(buf, np.ones(size[::-1], dtype=np.float32))
    return buf.getvalue()
