from pathlib import Path
from typing import Protocol

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
        **params,
    ) -> Path: ...
