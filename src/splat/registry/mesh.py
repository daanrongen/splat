"""Wiring seam for both mesh directions: `MESH_CATALOG` for gaussian-cloud
-> file export (still empty — see `ports/mesh.py`'s `MeshExporter` docstring
for why), and `MESH_PREDICTION_CATALOG` for image(+depth) -> mesh
prediction.
"""

from dataclasses import dataclass
from typing import Literal

from splat.domain.value_objects import ModelLicense
from splat.ports.mesh import MeshExporter, MeshPredictionBackend

Runtime = Literal["mlx", "coreml", "torch", "geometry"]


@dataclass(frozen=True)
class MeshModelDescriptor:
    name: str
    backend_cls: type[MeshExporter]


MESH_CATALOG: dict[str, MeshModelDescriptor] = {}


@dataclass(frozen=True)
class MeshPredictionModelDescriptor:
    name: str
    backend_cls: type[MeshPredictionBackend]
    hf_repo_id: str | None
    license: ModelLicense
    runtime: Runtime


def _build_prediction_catalog() -> dict[str, MeshPredictionModelDescriptor]:
    from splat.adapters.mesh.depth_heightfield import DepthHeightfieldBackend
    from splat.domain.value_objects import MIT

    return {
        "depth-heightfield": MeshPredictionModelDescriptor(
            name="depth-heightfield",
            backend_cls=DepthHeightfieldBackend,
            hf_repo_id=None,
            license=MIT,
            runtime="geometry",
        ),
    }


MESH_PREDICTION_CATALOG: dict[str, MeshPredictionModelDescriptor] = _build_prediction_catalog()
