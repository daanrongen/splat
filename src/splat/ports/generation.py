from pathlib import Path
from typing import Protocol

from splat.domain.value_objects import ModelLicense


class ImageGenerationBackend(Protocol):
    name: str
    license: ModelLicense

    def generate(
        self,
        prompt: str,
        *,
        output_path: Path,
        negative_prompt: str = "",
        steps: int | None = None,
        seed: int | None = None,
        **params,
    ) -> Path: ...
