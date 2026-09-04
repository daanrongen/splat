"""Model catalog for `splat mesh` image-to-mesh prediction backends."""

from dataclasses import dataclass
from typing import Literal

from splat.domain.contracts import Requirement, StageContract
from splat.domain.manifest import ManifestKind
from splat.domain.value_objects import ModelLicense
from splat.ports.mesh import MeshPredictionBackend

Runtime = Literal["mlx", "coreml", "torch"]

MESH_CONTRACT = StageContract(
    stage="mesh",
    inputs=(Requirement(name="image", any_of_tags=frozenset({"colorlike"}), max_count=None),),
    produces=ManifestKind.SHAPE_3D,
)


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
