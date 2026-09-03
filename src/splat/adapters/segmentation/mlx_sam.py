"""Automatic mask segmentation via Apple's MLX Segment Anything port
(vendored in `_vendor/mlx_sam`, see `_vendor/NOTICE.md`). Pure MLX/Metal —
no PyTorch, no CUDA.

Meta's original `facebook/sam-vit-base` checkpoint (Apache-2.0) is not
MLX-native, so its weights are converted once on first use (logic adapted
from mlx-examples' `convert.py`: a handful of conv weights need transposing
to MLX's channel-last layout) and cached under SPLAT_MODEL_CACHE_DIR.
"""

import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image

from splat.adapters.segmentation._vendor.mlx_sam import SamAutomaticMaskGenerator, sam
from splat.domain.image_space import Sticker
from splat.domain.value_objects import ModelLicense
from splat.paths import model_cache_dir

_TRANSPOSE_TO_NHWC = {
    "vision_encoder.patch_embed.projection.weight",
    "vision_encoder.neck.conv1.weight",
    "vision_encoder.neck.conv2.weight",
    "prompt_encoder.mask_embed.conv1.weight",
    "prompt_encoder.mask_embed.conv2.weight",
    "prompt_encoder.mask_embed.conv3.weight",
}
_TRANSPOSE_DECODER_UPSCALE = {
    "mask_decoder.upscale_conv1.weight",
    "mask_decoder.upscale_conv2.weight",
}


def _convert_to_mlx(hf_repo_id: str, mlx_dir: Path) -> None:
    if (mlx_dir / "model.safetensors").exists():
        return

    hf_path = Path(
        snapshot_download(repo_id=hf_repo_id, allow_patterns=["*.safetensors", "*.json"])
    )
    weights = mx.load(str(hf_path / "model.safetensors"))

    mlx_weights = {}
    for key, value in weights.items():
        if key in _TRANSPOSE_TO_NHWC:
            value = value.transpose(0, 2, 3, 1)
        elif key in _TRANSPOSE_DECODER_UPSCALE:
            value = value.transpose(1, 2, 3, 0)
        mlx_weights[key] = value

    mlx_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(mlx_dir / "model.safetensors"), mlx_weights)
    shutil.copy(hf_path / "config.json", mlx_dir / "config.json")


class MLXSamBackend:
    name = "mlx-sam"

    def __init__(self, *, hf_repo_id: str, license: ModelLicense, device: str = "auto") -> None:
        self._hf_repo_id = hf_repo_id
        self.license = license
        self._model = None

    def _mlx_weights_dir(self) -> Path:
        return model_cache_dir() / "mlx-sam" / self._hf_repo_id.replace("/", "--")

    def _load(self):
        if self._model is None:
            mlx_dir = self._mlx_weights_dir()
            _convert_to_mlx(self._hf_repo_id, mlx_dir)
            self._model = sam.load(str(mlx_dir))
        return self._model

    def segment(
        self, image_path: Path, *, max_stickers: int | None = None, **params
    ) -> list[Sticker]:
        model = self._load()
        generator = SamAutomaticMaskGenerator(model, **params)

        image = np.array(Image.open(image_path).convert("RGB"))
        masks = generator.generate(image)
        masks.sort(key=lambda m: m["area"], reverse=True)
        if max_stickers is not None:
            masks = masks[:max_stickers]

        stickers = []
        for mask_data in masks:
            mask = mask_data["segmentation"]
            x0, y0, w, h = (int(v) for v in mask_data["bbox"])
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, :3] = image[y0 : y0 + h, x0 : x0 + w]
            rgba[:, :, 3] = (mask[y0 : y0 + h, x0 : x0 + w] * 255).astype(np.uint8)
            stickers.append(
                Sticker(
                    rgba=rgba,
                    bbox=(x0, y0, w, h),
                    score=float(mask_data["predicted_iou"]),
                    area=int(mask_data["area"]),
                )
            )
        return stickers
