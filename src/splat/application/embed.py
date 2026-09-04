from pathlib import Path

import numpy as np

from splat.domain.errors import SplatDomainError
from splat.ports.embedding import EmbeddingBackend


def _as_embedding(vector: np.ndarray) -> np.ndarray:
    embedding = np.asarray(vector, dtype=np.float32).reshape(-1)
    if embedding.size == 0:
        raise SplatDomainError("Embedding backend returned an empty vector.")
    return embedding


class EmbedUseCase:
    def __init__(self, backend: EmbeddingBackend) -> None:
        self._backend = backend

    def image(self, image_path: Path, **params) -> np.ndarray:
        return _as_embedding(self._backend.embed_image(image_path, **params))

    def text(self, text: str, **params) -> np.ndarray:
        return _as_embedding(self._backend.embed_text(text, **params))
