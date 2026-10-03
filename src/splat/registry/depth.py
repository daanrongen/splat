from splat.domain.contracts import Requirement, StageContract
from splat.domain.manifest import ManifestKind
from splat.domain.value_objects import APACHE_2_0, APPLE_ASCL
from splat.registry.catalog import ModelDescriptor

DEPTH_CONTRACT = StageContract(
    stage="depth",
    inputs=(Requirement(name="image", any_of_tags=frozenset({"colorlike"}), max_count=None),),
    produces=ManifestKind.DEPTH_MAP,
)

DEPTH_CATALOG = {
    "depth-pro": ModelDescriptor(
        name="depth-pro",
        backend="splat.adapters.depth.depth_pro:DepthProBackend",
        hf_repo_ids=("apple/DepthPro-hf",),
        license=APPLE_ASCL,
        runtime="torch",
    ),
    "depth-anything-v2-coreml": ModelDescriptor(
        name="depth-anything-v2-coreml",
        backend="splat.adapters.depth.coreml_depth_anything_v2:CoreMLDepthAnythingV2Backend",
        hf_repo_ids=("apple/coreml-depth-anything-v2-small",),
        license=APACHE_2_0,
        runtime="coreml",
    ),
}
