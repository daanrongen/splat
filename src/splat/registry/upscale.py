from splat.domain.contracts import Requirement, StageContract
from splat.domain.manifest import ManifestKind
from splat.domain.value_objects import BSD_3_CLAUSE
from splat.registry.catalog import ModelDescriptor

UPSCALE_CONTRACT = StageContract(
    stage="upscale",
    inputs=(Requirement(name="image", any_of_tags=frozenset({"colorlike"}), max_count=None),),
    produces=ManifestKind.IMAGE,
)

UPSCALE_CATALOG = {
    "realesrgan-mlx": ModelDescriptor(
        name="realesrgan-mlx",
        backend="splat.adapters.upscale.realesrgan_mlx:RealESRGANMLXBackend",
        hf_repo_ids=("mlx-community/Real-ESRGAN-x2plus", "mlx-community/Real-ESRGAN-x4plus"),
        license=BSD_3_CLAUSE,
        runtime="mlx",
    ),
}
