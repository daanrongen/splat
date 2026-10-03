from splat.domain.contracts import Requirement, StageContract
from splat.domain.manifest import ManifestKind
from splat.domain.value_objects import APACHE_2_0
from splat.registry.catalog import ModelDescriptor

SEGMENT_CONTRACT = StageContract(
    stage="segment",
    inputs=(Requirement(name="image", any_of_tags=frozenset({"colorlike"}), max_count=None),),
    produces=ManifestKind.STICKER,
)

SEGMENTATION_CATALOG = {
    "sam-mlx": ModelDescriptor(
        name="sam-mlx",
        backend="splat.adapters.segment.mlx_sam:MLXSamBackend",
        hf_repo_ids=("facebook/sam-vit-base",),
        license=APACHE_2_0,
        runtime="mlx",
    ),
    "sam2-coreml": ModelDescriptor(
        name="sam2-coreml",
        backend="splat.adapters.segment.coreml_sam2:CoreMLSam2Backend",
        hf_repo_ids=("apple/coreml-sam2.1-tiny",),
        license=APACHE_2_0,
        runtime="coreml",
    ),
}
