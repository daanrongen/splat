from dataclasses import dataclass
from typing import Literal

from splat.domain.value_objects import ModelLicense
from splat.ports.reconstruction import ReconstructionBackend
from splat.registry._lazy import load_backend

Runtime = Literal["mlx", "torch"]


@dataclass(frozen=True)
class GaussianModelDescriptor:
    name: str
    backend: str
    hf_repo_id: str | None
    license: ModelLicense
    runtime: Runtime
    notes: str = ""

    @property
    def backend_cls(self) -> type[ReconstructionBackend]:
        return load_backend(self.backend)


def _build_catalog() -> dict[str, GaussianModelDescriptor]:
    from splat.domain.value_objects import APPLE_AMLR, MIT

    return {
        "mlx3d-capture": GaussianModelDescriptor(
            name="mlx3d-capture",
            backend="splat.adapters.gaussian.mlx3d_capture:MLX3DCaptureBackend",
            hf_repo_id=None,
            license=MIT,
            runtime="mlx",
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
            notes=(
                "Apple SHARP: single-image feed-forward 3DGS in one pass. Metric "
                "absolute scale, OpenCV/COLMAP convention. Research use only."
            ),
        ),
    }


GAUSSIAN_CATALOG: dict[str, GaussianModelDescriptor] = _build_catalog()
