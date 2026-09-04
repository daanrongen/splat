from dataclasses import dataclass
from typing import Literal

from splat.domain.value_objects import ModelLicense
from splat.ports.embedding import EmbeddingBackend

Runtime = Literal["mlx", "coreml", "torch"]


@dataclass(frozen=True)
class EmbeddingModelDescriptor:
    name: str
    backend_cls: type[EmbeddingBackend]
    hf_repo_id: str
    license: ModelLicense
    runtime: Runtime
    dimension: int
    normalized: bool
    notes: str = ""


def _build_catalog() -> dict[str, EmbeddingModelDescriptor]:
    from splat.adapters.embedding.mobileclip import MobileCLIPOpenCLIPBackend
    from splat.domain.value_objects import APPLE_AMLR

    return {
        "mobileclip2-s0": EmbeddingModelDescriptor(
            name="mobileclip2-s0",
            backend_cls=MobileCLIPOpenCLIPBackend,
            hf_repo_id="timm/MobileCLIP2-S0-OpenCLIP",
            license=APPLE_AMLR,
            runtime="torch",
            dimension=512,
            normalized=True,
            notes=(
                "MobileCLIP2-S0 OpenCLIP packaging of Apple's research-only MobileCLIP2 checkpoint."
            ),
        ),
    }


EMBEDDING_CATALOG: dict[str, EmbeddingModelDescriptor] = _build_catalog()
