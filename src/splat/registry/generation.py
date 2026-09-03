from dataclasses import dataclass
from typing import Literal

from splat.domain.value_objects import ModelLicense
from splat.ports.generation import ImageGenerationBackend

Runtime = Literal["mlx", "coreml", "torch"]


@dataclass(frozen=True)
class GenerationModelDescriptor:
    name: str
    backend_cls: type[ImageGenerationBackend]
    hf_repo_id: str
    sdxl: bool
    license: ModelLicense
    runtime: Runtime


def _build_catalog() -> dict[str, GenerationModelDescriptor]:
    from splat.adapters.generation.coreml_stable_diffusion import CoreMLStableDiffusionBackend
    from splat.adapters.generation.mlx_stable_diffusion import MLXStableDiffusionBackend
    from splat.domain.value_objects import OPENRAIL_M, SAI_NC_COMMUNITY

    return {
        "sdxl-turbo-mlx": GenerationModelDescriptor(
            name="sdxl-turbo-mlx",
            backend_cls=MLXStableDiffusionBackend,
            hf_repo_id="stabilityai/sdxl-turbo",
            sdxl=True,
            license=SAI_NC_COMMUNITY,
            runtime="mlx",
        ),
        "sd21-coreml": GenerationModelDescriptor(
            name="sd21-coreml",
            backend_cls=CoreMLStableDiffusionBackend,
            hf_repo_id="apple/coreml-stable-diffusion-2-1-base",
            sdxl=False,
            license=OPENRAIL_M,
            runtime="coreml",
        ),
    }


GENERATION_CATALOG: dict[str, GenerationModelDescriptor] = _build_catalog()
