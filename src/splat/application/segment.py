from pathlib import Path

from splat.domain.image_space import Sticker
from splat.ports.segmentation import SegmentationBackend


class SegmentUseCase:
    def __init__(self, backend: SegmentationBackend) -> None:
        self._backend = backend

    def execute(
        self, image_path: Path, *, max_stickers: int | None = None, **params
    ) -> list[Sticker]:
        return self._backend.segment(image_path, max_stickers=max_stickers, **params)
