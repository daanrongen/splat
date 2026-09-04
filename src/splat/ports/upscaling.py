from typing import Protocol

import numpy as np

from splat.domain.value_objects import ModelLicense


class UpscalingBackend(Protocol):
    name: str
    license: ModelLicense
    supported_factors: tuple[int, ...]

    def upscale(self, image: np.ndarray, *, factor: int, tile: int = 0, **params) -> np.ndarray: ...
