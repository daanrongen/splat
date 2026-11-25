from splat.domain.contracts import Requirement, StageContract
from splat.domain.manifest import ManifestKind
from splat.domain.value_objects import OPENRAIL_M, SAI_NC_COMMUNITY
from splat.registry.catalog import ModelDescriptor

_SDXL_TURBO_FILES = (
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "scheduler/scheduler_config.json",
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
    "tokenizer_2/vocab.json",
    "tokenizer_2/merges.txt",
)


def _coreml_files(prefix: str) -> tuple[str, ...]:
    packages = (
        f"original/packages/{prefix}_{part}.mlpackage/**"
        for part in ("text_encoder", "unet", "vae_decoder", "vae_encoder")
    )
    return ("original/compiled/vocab.json", "original/compiled/merges.txt", *packages)


_SD21_PREFIX = "Stable_Diffusion_version_stabilityai_stable-diffusion-2-1-base"
_SD15_PREFIX = "Stable_Diffusion_version_runwayml_stable-diffusion-v1-5"

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
        hf_files={"stabilityai/sdxl-turbo": _SDXL_TURBO_FILES},
        license=SAI_NC_COMMUNITY,
        runtime="mlx",
        backend_kwargs={"sdxl": True},
    ),
    "sd21-coreml": ModelDescriptor(
        name="sd21-coreml",
        backend="splat.adapters.diffuse.coreml_stable_diffusion:CoreMLStableDiffusionBackend",
        hf_repo_ids=("apple/coreml-stable-diffusion-2-1-base",),
        hf_files={"apple/coreml-stable-diffusion-2-1-base": _coreml_files(_SD21_PREFIX)},
        license=OPENRAIL_M,
        runtime="coreml",
        backend_kwargs={
            "sdxl": False,
            "package_prefix": _SD21_PREFIX,
        },
    ),
    "sd15-coreml": ModelDescriptor(
        name="sd15-coreml",
        backend="splat.adapters.diffuse.coreml_stable_diffusion:CoreMLStableDiffusionBackend",
        hf_repo_ids=("apple/coreml-stable-diffusion-v1-5-palettized",),
        hf_files={"apple/coreml-stable-diffusion-v1-5-palettized": _coreml_files(_SD15_PREFIX)},
        license=OPENRAIL_M,
        runtime="coreml",
        backend_kwargs={
            "sdxl": False,
            "package_prefix": _SD15_PREFIX,
        },
        notes="6-bit palettized SD 1.5, fixed 512x512, supports image-to-image",
    ),
}
