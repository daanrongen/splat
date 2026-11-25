from pathlib import Path

from splat.adapters.diffuse.mlx_stable_diffusion import MLXStableDiffusionBackend
from splat.domain.value_objects import SAI_NC_COMMUNITY

WEIGHTS = Path("/models/stabilityai--sdxl-turbo")


def _backend(*, sdxl: bool) -> MLXStableDiffusionBackend:
    return MLXStableDiffusionBackend(weights_paths=(WEIGHTS,), sdxl=sdxl, license=SAI_NC_COMMUNITY)


def test_sdxl_model_loads_from_the_weights_directory(mocker):
    xl = mocker.patch("splat.adapters.diffuse.mlx_stable_diffusion.StableDiffusionXL")

    _backend(sdxl=True)._load()

    xl.assert_called_once_with(str(WEIGHTS), float16=True)


def test_sd_model_loads_from_the_weights_directory(mocker):
    sd = mocker.patch("splat.adapters.diffuse.mlx_stable_diffusion.StableDiffusion")

    _backend(sdxl=False)._load()

    sd.assert_called_once_with(str(WEIGHTS), float16=True)


def test_model_is_loaded_once(mocker):
    xl = mocker.patch("splat.adapters.diffuse.mlx_stable_diffusion.StableDiffusionXL")
    backend = _backend(sdxl=True)

    backend._load()
    backend._load()

    xl.assert_called_once()
