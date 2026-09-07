"""Monocular metric depth estimation via Apple's DepthPro (`apple/DepthPro-hf`).

No MLX port of DepthPro exists yet, so this is the one adapter in the
project that still uses PyTorch — on `torch.device("mps")` it's still fully
Apple Silicon GPU-accelerated, just not through MLX. Revisit if/when an MLX
port appears.
"""

from pathlib import Path

import numpy as np
import torch
from transformers import DepthProForDepthEstimation, DepthProImageProcessor

from splat.adapters.formats.image import read_rgb
from splat.domain.image_space import DepthMap
from splat.domain.value_objects import ModelLicense


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DepthProBackend:
    name = "depth-pro"

    def __init__(self, *, hf_repo_id: str, license: ModelLicense, device: str = "auto") -> None:
        self._hf_repo_id = hf_repo_id
        self.license = license
        self._device = _resolve_device(device)
        self._model: DepthProForDepthEstimation | None = None
        self._processor: DepthProImageProcessor | None = None

    def _load(self) -> None:
        if self._model is None:
            self._processor = DepthProImageProcessor.from_pretrained(self._hf_repo_id)
            self._model = DepthProForDepthEstimation.from_pretrained(self._hf_repo_id).to(
                self._device
            )
            self._model.eval()

    def estimate(self, image_path: Path, **params) -> DepthMap:
        self._load()
        image = read_rgb(image_path)

        inputs = self._processor(images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)

        post_processed = self._processor.post_process_depth_estimation(
            outputs, target_sizes=[(image.shape[0], image.shape[1])]
        )[0]

        depth = post_processed["predicted_depth"].to("cpu").numpy().astype(np.float32)
        focal_length = post_processed.get("focal_length")
        field_of_view = post_processed.get("field_of_view")

        return DepthMap(
            depth=depth,
            focal_length_px=float(focal_length) if focal_length is not None else None,
            field_of_view_deg=float(field_of_view) if field_of_view is not None else None,
            metadata={"source_model": self._hf_repo_id, "device": self._device},
        )
