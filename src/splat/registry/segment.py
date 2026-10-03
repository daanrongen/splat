from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal

from splat.domain.contracts import Requirement, StageContract
from splat.domain.manifest import ManifestKind
from splat.domain.value_objects import ModelLicense

Runtime = Literal["mlx", "coreml", "torch"]

SEGMENT_CONTRACT = StageContract(
    stage="segment",
    inputs=(Requirement(name="image", any_of_tags=frozenset({"colorlike"}), max_count=None),),
    produces=ManifestKind.STICKER,
)


@dataclass(frozen=True)
class SegmentationModelDescriptor:
    name: str
    backend: str
    hf_repo_id: str
    license: ModelLicense
    runtime: Runtime

    @property
    def backend_cls(self) -> type[Any]:
        module_name, class_name = self.backend.rsplit(":", 1)
        return getattr(import_module(module_name), class_name)


def _build_catalog() -> dict[str, SegmentationModelDescriptor]:
    from splat.domain.value_objects import APACHE_2_0

    return {
        "sam-mlx": SegmentationModelDescriptor(
            name="sam-mlx",
            backend="splat.adapters.segment.mlx_sam:MLXSamBackend",
            hf_repo_id="facebook/sam-vit-base",
            license=APACHE_2_0,
            runtime="mlx",
        ),
        "sam2-coreml": SegmentationModelDescriptor(
            name="sam2-coreml",
            backend="splat.adapters.segment.coreml_sam2:CoreMLSam2Backend",
            hf_repo_id="apple/coreml-sam2.1-tiny",
            license=APACHE_2_0,
            runtime="coreml",
        ),
    }


SEGMENTATION_CATALOG: dict[str, SegmentationModelDescriptor] = _build_catalog()
