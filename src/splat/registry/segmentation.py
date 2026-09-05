from dataclasses import dataclass
from typing import Literal

from splat.domain.contracts import Requirement, StageContract
from splat.domain.manifest import ManifestKind
from splat.domain.value_objects import ModelLicense
from splat.ports.segmentation import SegmentationBackend

Runtime = Literal["mlx", "coreml", "torch"]

SEGMENT_CONTRACT = StageContract(
    stage="segment",
    inputs=(Requirement(name="image", any_of_tags=frozenset({"colorlike"}), max_count=None),),
    produces=ManifestKind.STICKER,
)


@dataclass(frozen=True)
class SegmentationModelDescriptor:
    name: str
    backend_cls: type[SegmentationBackend]
    hf_repo_id: str
    license: ModelLicense
    runtime: Runtime


def _build_catalog() -> dict[str, SegmentationModelDescriptor]:
    from splat.adapters.segmentation.coreml_sam2 import CoreMLSam2Backend
    from splat.adapters.segmentation.mlx_sam import MLXSamBackend
    from splat.domain.value_objects import APACHE_2_0

    return {
        "sam-mlx": SegmentationModelDescriptor(
            name="sam-mlx",
            backend_cls=MLXSamBackend,
            hf_repo_id="facebook/sam-vit-base",
            license=APACHE_2_0,
            runtime="mlx",
        ),
        "sam2-coreml": SegmentationModelDescriptor(
            name="sam2-coreml",
            backend_cls=CoreMLSam2Backend,
            hf_repo_id="apple/coreml-sam2.1-tiny",
            license=APACHE_2_0,
            runtime="coreml",
        ),
    }


SEGMENTATION_CATALOG: dict[str, SegmentationModelDescriptor] = _build_catalog()
