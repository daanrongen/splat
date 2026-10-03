from pathlib import Path
from typing import Protocol

import numpy as np

from splat.domain.value_objects import ModelLicense


class DiffusionBackend(Protocol):
    name: str
    license: ModelLicense

    def diffuse(
        self,
        prompt: str,
        *,
        output_path: Path,
        negative_prompt: str = "",
        steps: int | None = None,
        seed: int | None = None,
        image: np.ndarray | None = None,
        strength: float | None = None,
        width: int | None = None,
        height: int | None = None,
        **params,
    ) -> Path: ...
