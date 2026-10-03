from splat.domain.contracts import Requirement, StageContract
from splat.domain.manifest import ManifestKind
from splat.domain.value_objects import OPENRAIL_M, SAI_NC_COMMUNITY
from splat.registry.catalog import ModelDescriptor

DIFFUSE_CONTRACT = StageContract(
    stage="diffuse",
    inputs=(
        Requirement(name="image", any_of_tags=frozenset({"colorlike"}), min_count=0, max_count=1),
    ),
    produces=ManifestKind.IMAGE,
)

DIFFUSION_CATALOG = {
    "sdxl-turbo-mlx": ModelDescriptor(
        name="sdxl-turbo-mlx",
        backend="splat.adapters.diffuse.mlx_stable_diffusion:MLXStableDiffusionBackend",
        hf_repo_ids=("stabilityai/sdxl-turbo",),
        license=SAI_NC_COMMUNITY,
        runtime="mlx",
        backend_kwargs={"sdxl": True},
    ),
    "sd21-coreml": ModelDescriptor(
        name="sd21-coreml",
        backend="splat.adapters.diffuse.coreml_stable_diffusion:CoreMLStableDiffusionBackend",
        hf_repo_ids=("apple/coreml-stable-diffusion-2-1-base",),
        license=OPENRAIL_M,
        runtime="coreml",
        backend_kwargs={"sdxl": False},
    ),
}
