"""Wiring seam for the `MeshPredictionBackend` catalog `splat mesh` selects
from. `MeshExporter` (`splat tools extract.surface`) has no registry at
all — same as `ports/training.py`/`ports/rasterizer.py` — since it has zero
adapters to select between.
"""

from dataclasses import dataclass
from typing import Literal

from splat.domain.value_objects import ModelLicense
from splat.ports.mesh import MeshPredictionBackend

Runtime = Literal["mlx", "coreml", "torch"]


@dataclass(frozen=True)
class MeshPredictionModelDescriptor:
    name: str
    backend_cls: type[MeshPredictionBackend]
    hf_repo_id: str
    license: ModelLicense
    runtime: Runtime


def _build_catalog() -> dict[str, MeshPredictionModelDescriptor]:
    from splat.adapters.mesh.triposr import TripoSRBackend
    from splat.domain.value_objects import MIT

    return {
        "triposr": MeshPredictionModelDescriptor(
            name="triposr",
            backend_cls=TripoSRBackend,
            hf_repo_id="stabilityai/TripoSR",
            license=MIT,
            runtime="torch",
        ),
    }


MESH_PREDICTION_CATALOG: dict[str, MeshPredictionModelDescriptor] = _build_catalog()
