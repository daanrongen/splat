from dataclasses import dataclass
from typing import Literal

from splat.domain.contracts import Requirement, StageContract
from splat.domain.manifest import ManifestKind
from splat.domain.value_objects import ModelLicense
from splat.ports.diffusion import DiffusionBackend

Runtime = Literal["mlx", "coreml", "torch"]

DIFFUSE_CONTRACT = StageContract(
    stage="diffuse",
    inputs=(
        Requirement(name="image", any_of_tags=frozenset({"colorlike"}), min_count=0, max_count=1),
    ),
    produces=ManifestKind.IMAGE,
)


@dataclass(frozen=True)
class DiffusionModelDescriptor:
    name: str
    backend_cls: type[DiffusionBackend]
    hf_repo_id: str
    sdxl: bool
    license: ModelLicense
    runtime: Runtime


def _build_catalog() -> dict[str, DiffusionModelDescriptor]:
    from splat.adapters.diffuse.coreml_stable_diffusion import CoreMLStableDiffusionBackend
    from splat.adapters.diffuse.mlx_stable_diffusion import MLXStableDiffusionBackend
    from splat.domain.value_objects import OPENRAIL_M, SAI_NC_COMMUNITY

    return {
        "sdxl-turbo-mlx": DiffusionModelDescriptor(
            name="sdxl-turbo-mlx",
            backend_cls=MLXStableDiffusionBackend,
            hf_repo_id="stabilityai/sdxl-turbo",
            sdxl=True,
            license=SAI_NC_COMMUNITY,
            runtime="mlx",
        ),
        "sd21-coreml": DiffusionModelDescriptor(
            name="sd21-coreml",
            backend_cls=CoreMLStableDiffusionBackend,
            hf_repo_id="apple/coreml-stable-diffusion-2-1-base",
            sdxl=False,
            license=OPENRAIL_M,
            runtime="coreml",
        ),
    }


DIFFUSION_CATALOG: dict[str, DiffusionModelDescriptor] = _build_catalog()
