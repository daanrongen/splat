from splat.domain.contracts import Requirement, StageContract
from splat.domain.manifest import ManifestKind
from splat.domain.value_objects import APACHE_2_0, APPLE_AMLR
from splat.registry.catalog import ModelDescriptor

DEPTH_CONTRACT = StageContract(
    stage="depth",
    inputs=(Requirement(name="image", any_of_tags=frozenset({"colorlike"}), max_count=None),),
    produces=ManifestKind.DEPTH_MAP,
)

DEPTH_CATALOG = {
    "depth-pro-2": ModelDescriptor(
        name="depth-pro-2",
        backend="splat.adapters.depth.depth_pro_2:DepthPro2Backend",
        hf_repo_ids=("apple/DepthPro2-H", "apple/DepthPro2-focal"),
        hf_files={
            "apple/DepthPro2-H": ("config.json", "model.safetensors"),
            "apple/DepthPro2-focal": ("model.safetensors",),
        },
        license=APPLE_AMLR,
        runtime="torch",
    ),
    "depth-anything-v2-coreml": ModelDescriptor(
        name="depth-anything-v2-coreml",
        backend="splat.adapters.depth.coreml_depth_anything_v2:CoreMLDepthAnythingV2Backend",
        hf_repo_ids=("apple/coreml-depth-anything-v2-small",),
        hf_files={
            "apple/coreml-depth-anything-v2-small": ("DepthAnythingV2SmallF16.mlpackage/**",)
        },
        license=APACHE_2_0,
        runtime="coreml",
    ),
}
