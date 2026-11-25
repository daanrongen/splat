"""Monocular metric depth via Apple's Depth Pro 2 (H+ weights plus the focal estimator)."""

import math
from pathlib import Path

import numpy as np
import torch

from splat.adapters.formats.image import read_rgb
from splat.domain.image_space import DepthMap
from splat.domain.value_objects import ModelLicense


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DepthPro2Backend:
    name = "depth-pro-2"

    def __init__(
        self, *, hf_repo_id: str, focal_repo_id: str, license: ModelLicense, device: str = "auto"
    ) -> None:
        self._hf_repo_id = hf_repo_id
        self._focal_repo_id = focal_repo_id
        self.license = license
        self._device = _resolve_device(device)
        self._precision = torch.float32 if self._device == "cpu" else torch.bfloat16
        self._model = None
        self._transform = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from depth_pro_2 import DepthPro2
        from depth_pro_2.model import to_normalized_tensor
        from depth_pro_2.network.fov import FOVEstimator
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        model = DepthPro2.from_pretrained(self._hf_repo_id, precision=self._precision)
        model.fov = FOVEstimator(self._precision)
        weights = hf_hub_download(self._focal_repo_id, "model.safetensors")
        model.fov.load_state_dict(load_file(weights))
        self._model = model.to(self._device).eval()
        device = torch.device(self._device)
        self._transform = lambda image: to_normalized_tensor(image, device)

    def estimate(self, image_path: Path, **params) -> DepthMap:
        self._load()
        image = read_rgb(image_path)
        prediction = self._model.infer(self._transform(image))

        depth = prediction["depth"].float().cpu().numpy().astype(np.float32)
        focal = float(prediction["focallength_px"])
        fov = math.degrees(2 * math.atan(image.shape[1] / (2 * focal)))

        return DepthMap(
            depth=depth,
            focal_length_px=focal,
            field_of_view_deg=fov,
            metadata={
                "source_model": self._hf_repo_id,
                "device": self._device,
                "precision": str(self._precision).removeprefix("torch."),
            },
        )
