from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from splat.domain.errors import SplatDomainError

Interpolation = Literal["nearest", "linear", "cubic", "lanczos"]

_INTERPOLATION: dict[Interpolation, int] = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}


def _read_cv(path: Path, flags: int) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise SplatDomainError(f"Could not read image {str(path)!r}.")
    return image


def read_rgb(path: Path) -> np.ndarray:
    return cv2.cvtColor(_read_cv(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def read_rgb_or_rgba(path: Path) -> np.ndarray:
    image = _read_cv(path, cv2.IMREAD_UNCHANGED)
    return cv_to_rgb_or_rgba(image)


def decode_rgb_or_rgba(content: bytes) -> np.ndarray:
    encoded = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SplatDomainError("Could not decode image bytes.")
    return cv_to_rgb_or_rgba(image)


def cv_to_rgb_or_rgba(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize(
    image: np.ndarray, size: tuple[int, int], *, interpolation: Interpolation = "lanczos"
) -> np.ndarray:
    return cv2.resize(image, size, interpolation=_INTERPOLATION[interpolation])


def encode_png(image: np.ndarray) -> bytes:
    if image.ndim == 3 and image.shape[2] == 4:
        encoded_input = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    elif image.ndim == 3 and image.shape[2] == 3:
        encoded_input = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif image.ndim == 2:
        encoded_input = image
    else:
        raise SplatDomainError(f"Unsupported image shape for PNG encoding: {image.shape}.")

    ok, encoded = cv2.imencode(".png", encoded_input)
    if not ok:
        raise SplatDomainError("Could not encode image as PNG.")
    return encoded.tobytes()


def write_png(path: Path, image: np.ndarray) -> None:
    path.write_bytes(encode_png(image))
