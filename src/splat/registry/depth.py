from dataclasses import dataclass
from typing import Literal

from splat.domain.contracts import Requirement, StageContract
from splat.domain.manifest import ManifestKind
from splat.domain.value_objects import ModelLicense
from splat.ports.depth import DepthEstimationBackend

Runtime = Literal["mlx", "coreml", "torch"]

DEPTH_CONTRACT = StageContract(
    stage="depth",
    inputs=(Requirement(name="image", any_of_tags=frozenset({"colorlike"}), max_count=None),),
    produces=ManifestKind.DEPTH_MAP,
)


@dataclass(frozen=True)
class DepthModelDescriptor:
    name: str
    backend_cls: type[DepthEstimationBackend]
    hf_repo_id: str
    license: ModelLicense
    runtime: Runtime


def _build_catalog() -> dict[str, DepthModelDescriptor]:
    from splat.adapters.depth.coreml_depth_anything_v2 import CoreMLDepthAnythingV2Backend
    from splat.adapters.depth.depth_pro import DepthProBackend
    from splat.domain.value_objects import APACHE_2_0, APPLE_ASCL

    return {
        "depth-pro": DepthModelDescriptor(
            name="depth-pro",
            backend_cls=DepthProBackend,
            hf_repo_id="apple/DepthPro-hf",
            license=APPLE_ASCL,
            runtime="torch",
        ),
        "depth-anything-v2-coreml": DepthModelDescriptor(
            name="depth-anything-v2-coreml",
            backend_cls=CoreMLDepthAnythingV2Backend,
            hf_repo_id="apple/coreml-depth-anything-v2-small",
            license=APACHE_2_0,
            runtime="coreml",
        ),
    }


DEPTH_CATALOG: dict[str, DepthModelDescriptor] = _build_catalog()
