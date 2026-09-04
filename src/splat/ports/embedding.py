from pathlib import Path
from typing import Protocol

import numpy as np

from splat.domain.value_objects import ModelLicense


class EmbeddingBackend(Protocol):
    name: str
    license: ModelLicense

    def embed_image(self, image_path: Path, **params) -> np.ndarray: ...

    def embed_text(self, text: str, **params) -> np.ndarray: ...
