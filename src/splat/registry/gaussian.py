from dataclasses import dataclass
from typing import Literal

from splat.domain.value_objects import ModelLicense
from splat.ports.reconstruction import ReconstructionBackend

Runtime = Literal["mlx", "torch"]


@dataclass(frozen=True)
class GaussianModelDescriptor:
    name: str
    backend_cls: type[ReconstructionBackend]
    hf_repo_id: str | None
    license: ModelLicense
    runtime: Runtime
    notes: str = ""


def _build_catalog() -> dict[str, GaussianModelDescriptor]:
    # Imported lazily so a missing/optional adapter dependency can't break
    # every other command — only `splat models pull/gaussian --model ...`
    # needs the reconstruction adapters to actually import cleanly.
    from splat.adapters.gaussian.mlx3d_capture import MLX3DCaptureBackend
    from splat.adapters.gaussian.sharp import SharpBackend
    from splat.domain.value_objects import APPLE_AMLR, MIT

    return {
        "mlx3d-capture": GaussianModelDescriptor(
            name="mlx3d-capture",
            backend_cls=MLX3DCaptureBackend,
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
            backend_cls=SharpBackend,
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
