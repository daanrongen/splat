from dataclasses import dataclass
from pathlib import Path

import numpy as np

from splat.domain.errors import SplatDomainError
from splat.image_io import read_rgb_or_rgba, resize
from splat.ports.upscaling import UpscalingBackend


@dataclass(frozen=True)
class UpscaleResult:
    image: np.ndarray
    source_width: int
    source_height: int
    output_width: int
    output_height: int


class UpscaleUseCase:
    def __init__(self, backend: UpscalingBackend) -> None:
        self._backend = backend

    def execute(self, image_path: Path, *, factor: int, tile: int = 0, **params) -> UpscaleResult:
        if factor not in self._backend.supported_factors:
            available = ", ".join(str(f) for f in self._backend.supported_factors)
            raise SplatDomainError(f"{self._backend.name} supports --factor {available}.")

        image = read_rgb_or_rgba(image_path)
        if image.ndim == 3 and image.shape[2] == 4:
            rgb = image[:, :, :3]
            alpha = image[:, :, 3]
            upscaled_rgb = self._backend.upscale(rgb, factor=factor, tile=tile, **params)
            output_size = (upscaled_rgb.shape[1], upscaled_rgb.shape[0])
            upscaled_alpha = resize(alpha, output_size, interpolation="lanczos")
            output = np.dstack([upscaled_rgb, upscaled_alpha]).astype(np.uint8)
        else:
            output = self._backend.upscale(image, factor=factor, tile=tile, **params)

        return UpscaleResult(
            image=output,
            source_width=image.shape[1],
            source_height=image.shape[0],
            output_width=output.shape[1],
            output_height=output.shape[0],
        )
