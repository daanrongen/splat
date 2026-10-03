from splat.domain.contracts import Requirement, StageContract
from splat.domain.manifest import ManifestKind
from splat.domain.value_objects import APPLE_AMLR
from splat.registry.catalog import ModelDescriptor

# Only the image branch fits a StageContract — embed's image-XOR-text choice
# stays bespoke handler logic (see handlers/embed.py).
EMBED_IMAGE_CONTRACT = StageContract(
    stage="embed",
    inputs=(
        Requirement(name="image", any_of_tags=frozenset({"colorlike", "text"}), max_count=None),
    ),
    produces=ManifestKind.EMBEDDING,
)

EMBEDDING_CATALOG = {
    "mobileclip2-s0": ModelDescriptor(
        name="mobileclip2-s0",
        backend="splat.adapters.embed.mobileclip:MobileCLIPOpenCLIPBackend",
        hf_repo_ids=("timm/MobileCLIP2-S0-OpenCLIP",),
        license=APPLE_AMLR,
        runtime="torch",
        dimension=512,
        normalized=True,
        notes="MobileCLIP2-S0 OpenCLIP packaging of Apple's research-only MobileCLIP2 checkpoint.",
    ),
}
