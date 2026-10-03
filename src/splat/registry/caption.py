from splat.domain.contracts import Requirement, StageContract
from splat.domain.manifest import ManifestKind
from splat.domain.value_objects import APPLE_AMLR
from splat.registry.catalog import ModelDescriptor

CAPTION_CONTRACT = StageContract(
    stage="caption",
    inputs=(Requirement(name="image", any_of_tags=frozenset({"colorlike"}), max_count=None),),
    produces=ManifestKind.CAPTION,
)

CAPTION_CATALOG = {
    "fastvlm-0.5b": ModelDescriptor(
        name="fastvlm-0.5b",
        backend="splat.adapters.caption.fastvlm_transformers:FastVLMTransformersBackend",
        hf_repo_ids=("apple/FastVLM-0.5B",),
        license=APPLE_AMLR,
        runtime="torch",
    ),
}
