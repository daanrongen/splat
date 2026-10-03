from dataclasses import dataclass
from typing import Literal

from splat.domain.contracts import Requirement, StageContract
from splat.domain.manifest import ManifestKind
from splat.domain.value_objects import ModelLicense
from splat.ports.depth import DepthEstimationBackend
from splat.registry._lazy import load_backend

Runtime = Literal["mlx", "coreml", "torch"]

DEPTH_CONTRACT = StageContract(
    stage="depth",
    inputs=(Requirement(name="image", any_of_tags=frozenset({"colorlike"}), max_count=None),),
    produces=ManifestKind.DEPTH_MAP,
)


@dataclass(frozen=True)
class DepthModelDescriptor:
    name: str
    backend: str
    hf_repo_id: str
    license: ModelLicense
    runtime: Runtime

    @property
    def backend_cls(self) -> type[DepthEstimationBackend]:
        return load_backend(self.backend)


def _build_catalog() -> dict[str, DepthModelDescriptor]:
    from splat.domain.value_objects import APACHE_2_0, APPLE_ASCL

    return {
        "depth-pro": DepthModelDescriptor(
            name="depth-pro",
            backend="splat.adapters.depth.depth_pro:DepthProBackend",
            hf_repo_id="apple/DepthPro-hf",
            license=APPLE_ASCL,
            runtime="torch",
        ),
        "depth-anything-v2-coreml": DepthModelDescriptor(
            name="depth-anything-v2-coreml",
            backend="splat.adapters.depth.coreml_depth_anything_v2:CoreMLDepthAnythingV2Backend",
            hf_repo_id="apple/coreml-depth-anything-v2-small",
            license=APACHE_2_0,
            runtime="coreml",
        ),
    }


DEPTH_CATALOG: dict[str, DepthModelDescriptor] = _build_catalog()
