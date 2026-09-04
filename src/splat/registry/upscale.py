from dataclasses import dataclass
from typing import Literal

from splat.domain.contracts import Requirement, StageContract
from splat.domain.manifest import ManifestKind
from splat.domain.value_objects import BSD_3_CLAUSE, ModelLicense
from splat.ports.upscaling import UpscalingBackend

Runtime = Literal["mlx", "coreml", "torch"]

UPSCALE_CONTRACT = StageContract(
    stage="upscale",
    inputs=(Requirement(name="image", any_of_tags=frozenset({"colorlike"}), max_count=None),),
    produces=ManifestKind.IMAGE,
)


@dataclass(frozen=True)
class UpscaleModelDescriptor:
    name: str
    backend_cls: type[UpscalingBackend]
    hf_repo_ids: dict[int, str]
    license: ModelLicense
    runtime: Runtime
    supported_factors: tuple[int, ...]

    @property
    def hf_repo_id(self) -> str:
        return ", ".join(self.hf_repo_ids[factor] for factor in self.supported_factors)


def _build_catalog() -> dict[str, UpscaleModelDescriptor]:
    from splat.adapters.upscaling.realesrgan_mlx import RealESRGANMLXBackend

    return {
        "realesrgan-mlx": UpscaleModelDescriptor(
            name="realesrgan-mlx",
            backend_cls=RealESRGANMLXBackend,
            hf_repo_ids={
                2: "mlx-community/Real-ESRGAN-x2plus",
                4: "mlx-community/Real-ESRGAN-x4plus",
            },
            license=BSD_3_CLAUSE,
            runtime="mlx",
            supported_factors=(2, 4),
        ),
    }


UPSCALE_CATALOG: dict[str, UpscaleModelDescriptor] = _build_catalog()
