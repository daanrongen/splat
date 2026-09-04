from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np

from splat.domain.errors import SplatDomainError
from splat.domain.value_objects import ModelLicense


def _resolve_device(device: str) -> str:
    if device != "auto":
        if device not in ("cpu", "mps"):
            raise SplatDomainError("MobileCLIP embedding device must be 'auto', 'cpu', or 'mps'.")
        return device

    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class MobileCLIPOpenCLIPBackend:
    name = "mobileclip2-s0"

    def __init__(self, *, hf_repo_id: str, license: ModelLicense, device: str = "auto") -> None:
        self._hf_repo_id = hf_repo_id
        self.license = license
        self._device = _resolve_device(device)
        self._model: Any | None = None
        self._preprocess: Any | None = None
        self._tokenizer: Any | None = None

    def _load(self) -> None:
        if self._model is not None:
            return

        import open_clip
        from timm.utils import reparameterize_model

        model_name = f"hf-hub:{self._hf_repo_id}"
        model, _, preprocess = open_clip.create_model_and_transforms(model_name)
        model.eval()
        with suppress(Exception):
            model = reparameterize_model(model)
        self._model = model.to(self._device)
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(model_name)

    def _normalize(self, features: Any) -> np.ndarray:
        import torch

        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        if features.ndim == 2:
            features = features[0]
        return features.detach().to("cpu", dtype=torch.float32).numpy().astype(np.float32)

    def embed_image(self, image_path: Path, **params) -> np.ndarray:
        self._load()

        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        tensor = self._preprocess(image).unsqueeze(0).to(self._device)
        with torch.no_grad(), torch.amp.autocast(self._device, enabled=self._device != "cpu"):
            features = self._model.encode_image(tensor)
        return self._normalize(features)

    def embed_text(self, text: str, **params) -> np.ndarray:
        self._load()

        import torch

        tokens = self._tokenizer([text]).to(self._device)
        with torch.no_grad(), torch.amp.autocast(self._device, enabled=self._device != "cpu"):
            features = self._model.encode_text(tokens)
        return self._normalize(features)
