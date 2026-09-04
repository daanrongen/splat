import numpy as np

from splat.domain.value_objects import ModelLicense

_VARIANTS = {
    2: "RealESRGAN_x2plus",
    4: "RealESRGAN_x4plus",
}


class RealESRGANMLXBackend:
    name = "realesrgan-mlx"
    supported_factors = tuple(sorted(_VARIANTS))

    def __init__(self, *, license: ModelLicense) -> None:
        self.license = license
        self._upsamplers = {}

    def variant_for_factor(self, factor: int) -> str:
        return _VARIANTS[factor]

    def _load(self, *, factor: int, tile: int):
        key = (factor, tile)
        if key not in self._upsamplers:
            from realesrgan_mlx.pipeline_mlx import make_upsampler

            self._upsamplers[key] = make_upsampler(
                variant=self.variant_for_factor(factor),
                tile=tile,
            )
        return self._upsamplers[key]

    def upscale(self, image: np.ndarray, *, factor: int, tile: int = 0, **params) -> np.ndarray:
        upsampler = self._load(factor=factor, tile=tile)
        output, _mode = upsampler.enhance(image, outscale=None)
        return output.astype(np.uint8, copy=False)
