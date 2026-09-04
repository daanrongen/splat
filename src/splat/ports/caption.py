from pathlib import Path
from typing import Protocol

from splat.domain.value_objects import ModelLicense


class CaptioningBackend(Protocol):
    name: str
    license: ModelLicense

    def caption(
        self,
        image_path: Path,
        *,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **params,
    ) -> str: ...
