from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal

from splat.domain.value_objects import ModelLicense

Runtime = Literal["mlx", "torch"]


@dataclass(frozen=True)
class GaussianModelDescriptor:
    name: str
    backend: str
    hf_repo_id: str | None
    license: ModelLicense
    runtime: Runtime
    min_images: int
    max_images: int | None
    notes: str = ""

    @property
    def backend_cls(self) -> type[Any]:
        module_name, class_name = self.backend.rsplit(":", 1)
        return getattr(import_module(module_name), class_name)


def _build_catalog() -> dict[str, GaussianModelDescriptor]:
    from splat.domain.value_objects import APPLE_AMLR, MIT

    return {
        "mlx3d-capture": GaussianModelDescriptor(
            name="mlx3d-capture",
            backend="splat.adapters.gaussian.mlx3d_capture:MLX3DCaptureBackend",
            hf_repo_id=None,
            license=MIT,
            runtime="mlx",
            min_images=3,
            max_images=None,
            notes=(
                "Local Apple Silicon backend using mlx3d's optimization-based capture "
                "pipeline; requires 3+ photos or frames."
            ),
        ),
        "sharp": GaussianModelDescriptor(
            name="sharp",
            backend="splat.adapters.gaussian.sharp:SharpBackend",
            hf_repo_id="apple/Sharp",
            license=APPLE_AMLR,
            runtime="torch",
            min_images=1,
            max_images=1,
            notes=(
                "Apple SHARP: single-image feed-forward 3DGS in one pass. Metric "
                "absolute scale, OpenCV/COLMAP convention. Research use only."
            ),
        ),
    }


GAUSSIAN_CATALOG: dict[str, GaussianModelDescriptor] = _build_catalog()
