from pathlib import Path
from typing import Protocol

from splat.domain.image_space import Sticker
from splat.domain.value_objects import ModelLicense


class SegmentationBackend(Protocol):
    name: str
    license: ModelLicense

    def segment(
        self, image_path: Path, *, max_stickers: int | None = None, **params
    ) -> list[Sticker]: ...
