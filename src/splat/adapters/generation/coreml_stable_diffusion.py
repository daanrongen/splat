"""Text-to-image generation via Apple's official CoreML Stable Diffusion
conversion (`apple/coreml-stable-diffusion-2-1-base`), driven directly
through `coremltools` rather than Apple's `python_coreml_stable_diffusion`
reference package — that package requires torch + diffusers==0.30.2 +
transformers==4.44.2 (pinned old), which would conflict with this project's
`transformers` (used by the DepthPro adapter) without actually avoiding
torch. This reuses the vendored MLX Stable Diffusion's tokenizer (pure
regex BPE, no mlx dependency) and a NumPy port of its Euler sampler math,
calling the three CoreML .mlpackage files (text_encoder, unet, vae_decoder)
directly. I/O tensor names/shapes were confirmed against the real model
specs, not guessed — see coremltools' `load_spec()` on each package.
"""

import json
from pathlib import Path

import coremltools as ct
import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image

from splat.adapters.generation._vendor.mlx_stable_diffusion.tokenizer import Tokenizer
from splat.domain.value_objects import ModelLicense
from splat.paths import model_cache_dir

_PACKAGE_PREFIX = "Stable_Diffusion_version_stabilityai_stable-diffusion-2-1-base"
_SUBFOLDER = "original/packages"
_TOKENIZER_SUBFOLDER = "original/compiled"
_VAE_SCALING_FACTOR = 0.18215
_BETA_START, _BETA_END, _NUM_TRAIN_STEPS = 0.00085, 0.012, 1000
_MAX_TOKENS = 77


def _scaled_linear_sigmas() -> np.ndarray:
    """Same noise schedule as the vendored MLX sampler (SD's standard
    scaled-linear beta schedule) — reimplemented in NumPy."""
    betas = np.linspace(_BETA_START**0.5, _BETA_END**0.5, _NUM_TRAIN_STEPS) ** 2
    alphas_cumprod = np.cumprod(1 - betas)
    return np.concatenate([[0.0], np.sqrt((1 - alphas_cumprod) / alphas_cumprod)])


def _euler_timesteps(sigmas: np.ndarray, num_steps: int) -> list[tuple[float, float]]:
    steps = np.linspace(len(sigmas) - 1, 0, num_steps + 1)
    return list(zip(steps.tolist(), steps[1:].tolist(), strict=False))


def _sigma_at(sigmas: np.ndarray, t: float) -> float:
    lo = int(t)
    hi = min(lo + 1, len(sigmas) - 1)
    frac = t - lo
    return float(sigmas[lo] * (1 - frac) + sigmas[hi] * frac)


class CoreMLStableDiffusionBackend:
    name = "coreml-stable-diffusion"

    def __init__(
        self, *, hf_repo_id: str, sdxl: bool, license: ModelLicense, device: str = "auto"
    ) -> None:
        if sdxl:
            raise NotImplementedError("CoreMLStableDiffusionBackend only supports SD2.1-base.")
        self._hf_repo_id = hf_repo_id
        self.license = license
        self._compute_unit = (
            ct.ComputeUnit.CPU_ONLY if device == "cpu" else ct.ComputeUnit.CPU_AND_GPU
        )
        self._tokenizer: Tokenizer | None = None
        self._text_encoder = None
        self._unet = None
        self._vae_decoder = None

    def _package_path(self, component: str) -> str:
        return f"{_SUBFOLDER}/{_PACKAGE_PREFIX}_{component}.mlpackage"

    def _local_dir(self) -> Path:
        return model_cache_dir() / "coreml-stable-diffusion" / self._hf_repo_id.replace("/", "--")

    def _load(self) -> None:
        if self._unet is not None:
            return

        components = ["text_encoder", "unet", "vae_decoder"]
        local_dir = self._local_dir()
        have_tokenizer = (local_dir / _TOKENIZER_SUBFOLDER / "vocab.json").exists()
        have_packages = all((local_dir / self._package_path(c)).exists() for c in components)
        if not (have_tokenizer and have_packages):
            # local_dir= materializes real files (reusing already-downloaded blobs via
            # hardlink, no re-fetch) instead of the default symlinked cache — the
            # CoreML compiler cannot stage a symlinked weight.bin during compilation.
            snapshot_download(
                self._hf_repo_id,
                local_dir=local_dir,
                allow_patterns=[
                    f"{_TOKENIZER_SUBFOLDER}/vocab.json",
                    f"{_TOKENIZER_SUBFOLDER}/merges.txt",
                    *[f"{self._package_path(c)}/**" for c in components],
                ],
            )

        vocab = json.loads((local_dir / _TOKENIZER_SUBFOLDER / "vocab.json").read_text())
        merges_text = (local_dir / _TOKENIZER_SUBFOLDER / "merges.txt").read_text()
        bpe_merges = merges_text.strip().split("\n")[1 : 49152 - 256 - 2 + 1]
        bpe_ranks = dict(map(reversed, enumerate(tuple(m.split()) for m in bpe_merges)))
        self._tokenizer = Tokenizer(bpe_ranks, vocab)

        self._text_encoder = ct.models.MLModel(
            str(local_dir / self._package_path("text_encoder")), compute_units=self._compute_unit
        )
        self._unet = ct.models.MLModel(
            str(local_dir / self._package_path("unet")), compute_units=self._compute_unit
        )
        self._vae_decoder = ct.models.MLModel(
            str(local_dir / self._package_path("vae_decoder")), compute_units=self._compute_unit
        )

    def _encode(self, text: str) -> np.ndarray:
        tokens = self._tokenizer.tokenize(text)
        pad_id = self._tokenizer.eos_token
        tokens = (tokens + [pad_id] * _MAX_TOKENS)[:_MAX_TOKENS]
        input_ids = np.array([tokens], dtype=np.float32)
        out = self._text_encoder.predict({"input_ids": input_ids})
        hidden = out["last_hidden_state"]  # (1, 77, 1024)
        return hidden.transpose(0, 2, 1)[:, :, None, :]  # (1, 1024, 1, 77) — unet's expected layout

    def generate(
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
        self._load()
        steps = steps or 25
        cfg_weight = cfg_weight if cfg_weight is not None else 7.5
        rng = np.random.default_rng(seed)

        uncond = self._encode(negative_prompt or "")
        cond = self._encode(prompt)
        encoder_hidden_states = np.concatenate([uncond, cond], axis=0).astype(np.float16)

        sigmas = _scaled_linear_sigmas()
        timesteps = _euler_timesteps(sigmas, steps)

        latents = rng.standard_normal((1, 4, 64, 64)).astype(np.float32)
        latents = latents * sigmas[-1] / np.sqrt(sigmas[-1] ** 2 + 1)

        for t, t_prev in timesteps:
            sigma = _sigma_at(sigmas, t)
            sigma_prev = _sigma_at(sigmas, t_prev)

            sample = np.concatenate([latents, latents], axis=0).astype(np.float16)
            timestep_arr = np.array([t, t], dtype=np.float16)

            unet_out = self._unet.predict(
                {
                    "sample": sample,
                    "timestep": timestep_arr,
                    "encoder_hidden_states": encoder_hidden_states,
                }
            )
            noise_pred = unet_out["noise_pred"].astype(np.float32)
            eps_uncond, eps_cond = noise_pred[0:1], noise_pred[1:2]
            eps = eps_uncond + cfg_weight * (eps_cond - eps_uncond)

            x_t_prev = np.sqrt(sigma**2 + 1) * latents + eps * (sigma_prev - sigma)
            latents = x_t_prev / np.sqrt(sigma_prev**2 + 1)

        z = (latents / _VAE_SCALING_FACTOR).astype(np.float16)
        image = self._vae_decoder.predict({"z": z})["image"]  # (1, 3, 512, 512), roughly [-1, 1]

        image = np.clip(image[0].transpose(1, 2, 0) / 2 + 0.5, 0, 1)
        Image.fromarray((image * 255).astype(np.uint8)).save(output_path)
        return output_path
