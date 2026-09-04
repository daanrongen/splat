"""Text-to-image generation via Apple's MLX Stable Diffusion port (vendored
in `_vendor/mlx_stable_diffusion`, see `_vendor/NOTICE.md`). Pure MLX/Metal
— no PyTorch, no CUDA. Weights download from HuggingFace Hub on first use
into the standard HF cache (HF_HOME, set via mise.toml)."""

from pathlib import Path

import mlx.core as mx
import numpy as np

from splat.adapters.diffusion._vendor.mlx_stable_diffusion import (
    StableDiffusion,
    StableDiffusionXL,
)
from splat.domain.value_objects import ModelLicense
from splat.image_io import write_png

# The vendored loader only recognizes repo ids it has an explicit path-map
# entry for (see _vendor/mlx_stable_diffusion/model_io.py's `_MODELS`) —
# "stabilityai/sdxl-turbo" is one of the two it ships with, so no patching
# is needed here.


class MLXStableDiffusionBackend:
    name = "mlx-stable-diffusion"

    def __init__(
        self, *, hf_repo_id: str, sdxl: bool, license: ModelLicense, device: str = "auto"
    ) -> None:
        self._hf_repo_id = hf_repo_id
        self._sdxl = sdxl
        self.license = license
        self._model: StableDiffusion | StableDiffusionXL | None = None

    def _load(self) -> StableDiffusion:
        if self._model is None:
            cls = StableDiffusionXL if self._sdxl else StableDiffusion
            self._model = cls(self._hf_repo_id, float16=True)
        return self._model

    def diffuse(
        self,
        prompt: str,
        *,
        output_path: Path,
        negative_prompt: str = "",
        steps: int | None = None,
        seed: int | None = None,
        cfg_weight: float | None = None,
        **params,
    ) -> Path:
        model = self._load()
        default_steps = 2 if self._sdxl else 50
        default_cfg = 0.0 if self._sdxl else 7.5

        latents = None
        for x_t in model.generate_latents(
            prompt,
            n_images=1,
            num_steps=steps or default_steps,
            cfg_weight=cfg_weight if cfg_weight is not None else default_cfg,
            negative_text=negative_prompt,
            seed=seed,
        ):
            mx.eval(x_t)
            latents = x_t

        decoded = model.decode(latents)
        mx.eval(decoded)

        image_array = np.array((mx.clip(decoded[0], 0, 1) * 255).astype(mx.uint8))
        write_png(output_path, image_array)
        return output_path
