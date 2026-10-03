from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal

from splat.domain.contracts import Requirement, StageContract
from splat.domain.manifest import ManifestKind
from splat.domain.value_objects import ModelLicense

Runtime = Literal["mlx", "coreml", "torch"]

CAPTION_CONTRACT = StageContract(
    stage="caption",
    inputs=(Requirement(name="image", any_of_tags=frozenset({"colorlike"}), max_count=None),),
    produces=ManifestKind.CAPTION,
)


@dataclass(frozen=True)
class CaptionModelDescriptor:
    name: str
    backend: str
    hf_repo_id: str
    license: ModelLicense
    runtime: Runtime

    @property
    def backend_cls(self) -> type[Any]:
        module_name, class_name = self.backend.rsplit(":", 1)
        return getattr(import_module(module_name), class_name)


def _build_catalog() -> dict[str, CaptionModelDescriptor]:
    from splat.domain.value_objects import APPLE_AMLR

    return {
        "fastvlm-0.5b": CaptionModelDescriptor(
            name="fastvlm-0.5b",
            backend="splat.adapters.caption.fastvlm_transformers:FastVLMTransformersBackend",
            hf_repo_id="apple/FastVLM-0.5B",
            license=APPLE_AMLR,
            runtime="torch",
        ),
    }


CAPTION_CATALOG: dict[str, CaptionModelDescriptor] = _build_catalog()
