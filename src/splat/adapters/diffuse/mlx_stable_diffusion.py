"""Text-to-image generation via Apple's MLX Stable Diffusion port (vendored
in `_vendor/mlx_stable_diffusion`, see `_vendor/NOTICE.md`). Pure MLX/Metal
— no PyTorch, no CUDA. Weights download from HuggingFace Hub on first use
into the standard HF cache (HF_HOME, set via mise.toml)."""

from pathlib import Path

import mlx.core as mx
import numpy as np

from splat.adapters.diffuse._vendor.mlx_stable_diffusion import (
    StableDiffusion,
    StableDiffusionXL,
)
from splat.adapters.formats.image import resize, write_png
from splat.domain.errors import SplatDomainError
from splat.domain.value_objects import ModelLicense

_DEFAULT_SIZE = 512
_DEFAULT_STRENGTH = 0.7
# VAE downsamples 8x and the UNet 8x again, so dims must be multiples of 64
_SIZE_MULTIPLE = 64


def _resolve_size(width: int | None, height: int | None) -> tuple[int, int]:
    width = width if width is not None else _DEFAULT_SIZE
    height = height if height is not None else _DEFAULT_SIZE
    if width <= 0 or height <= 0 or width % _SIZE_MULTIPLE or height % _SIZE_MULTIPLE:
        raise SplatDomainError(
            f"--width/--height must be positive multiples of {_SIZE_MULTIPLE} "
            f"(got {width}x{height})."
        )
    return width, height


def _encode_init_image(
    model: StableDiffusion, image: np.ndarray, size: tuple[int, int]
) -> mx.array:
    resized = resize(image[..., :3], size)  # cv2 size is (W, H)
    normalized = resized.astype(np.float32) / 127.5 - 1.0
    return mx.array(normalized, dtype=model.dtype)


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
        image: np.ndarray | None = None,
        strength: float | None = None,
        width: int | None = None,
        height: int | None = None,
        **params,
    ) -> Path:
        model = self._load()
        default_steps = 2 if self._sdxl else 50
        default_cfg = 0.0 if self._sdxl else 7.5
        cfg_weight = cfg_weight if cfg_weight is not None else default_cfg
        width, height = _resolve_size(width, height)

        if image is not None:
            latents_iter = model.generate_latents_from_image(
                _encode_init_image(model, image, (width, height)),
                prompt,
                n_images=1,
                strength=strength if strength is not None else _DEFAULT_STRENGTH,
                num_steps=steps or default_steps,
                cfg_weight=cfg_weight,
                negative_text=negative_prompt,
                seed=seed,
            )
        else:
            latents_iter = model.generate_latents(
                prompt,
                n_images=1,
                num_steps=steps or default_steps,
                cfg_weight=cfg_weight,
                negative_text=negative_prompt,
                latent_size=(height // 8, width // 8),
                seed=seed,
            )

        latents = None
        for x_t in latents_iter:
            mx.eval(x_t)
            latents = x_t

        decoded = model.decode(latents)
        mx.eval(decoded)

        image_array = np.array((mx.clip(decoded[0], 0, 1) * 255).astype(mx.uint8))
        write_png(output_path, image_array)
        return output_path
