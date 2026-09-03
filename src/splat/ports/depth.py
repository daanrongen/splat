from pathlib import Path
from typing import Protocol

from splat.domain.image_space import DepthMap
from splat.domain.value_objects import ModelLicense


class DepthEstimationBackend(Protocol):
    name: str
    license: ModelLicense

    def estimate(self, image_path: Path, **params) -> DepthMap: ...
