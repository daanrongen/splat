from pathlib import Path

import numpy as np

from splat.image_io import encode_png, write_png


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
