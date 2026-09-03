from dataclasses import dataclass
from typing import Literal

from splat.domain.value_objects import ModelLicense
from splat.ports.depth import DepthEstimationBackend

Runtime = Literal["mlx", "coreml", "torch"]


@dataclass(frozen=True)
class DepthModelDescriptor:
    name: str
    backend_cls: type[DepthEstimationBackend]
    hf_repo_id: str
    license: ModelLicense
    runtime: Runtime


def _build_catalog() -> dict[str, DepthModelDescriptor]:
    from splat.adapters.depth.depth_pro import DepthProBackend
    from splat.domain.value_objects import APPLE_ASCL

    return {
        "depth-pro": DepthModelDescriptor(
            name="depth-pro",
            backend_cls=DepthProBackend,
            hf_repo_id="apple/DepthPro-hf",
            license=APPLE_ASCL,
            runtime="torch",
        ),
    }


DEPTH_CATALOG: dict[str, DepthModelDescriptor] = _build_catalog()
