import numpy as np

from splat.adapters.upscale.realesrgan_mlx import RealESRGANMLXBackend
from splat.domain.value_objects import BSD_3_CLAUSE


class FakeUpsampler:
    def __init__(self) -> None:
        self.calls = []

    def enhance(self, image, *, outscale=None):
        self.calls.append((image, outscale))
        return np.repeat(np.repeat(image, 2, axis=0), 2, axis=1), "RGB"


def test_variant_for_factor():
    backend = RealESRGANMLXBackend(license=BSD_3_CLAUSE)

    assert backend.variant_for_factor(2) == "RealESRGAN_x2plus"
    assert backend.variant_for_factor(4) == "RealESRGAN_x4plus"


def test_upscale_uses_native_variant_and_tile(mocker):
    fake = FakeUpsampler()
    make_upsampler = mocker.patch("realesrgan_mlx.pipeline_mlx.make_upsampler", return_value=fake)
    backend = RealESRGANMLXBackend(license=BSD_3_CLAUSE)
    image = np.zeros((2, 3, 3), dtype=np.uint8)

    result = backend.upscale(image, factor=2, tile=128)

    make_upsampler.assert_called_once_with(variant="RealESRGAN_x2plus", tile=128)
    assert fake.calls[0][1] is None
    assert result.shape == (4, 6, 3)
