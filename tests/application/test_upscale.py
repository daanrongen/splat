import numpy as np
import pytest

from splat.adapters.cache.filesystem import FilesystemManifestRepository
from splat.adapters.formats.image import read_rgb_or_rgba
from splat.application.pipeline import run_upscale
from splat.application.upscale import UpscaleUseCase
from splat.domain.errors import SplatDomainError
from splat.domain.manifest import ManifestKind
from splat.domain.value_objects import MIT
from tests.image_helpers import sample_rgba, write_sample_png


class FakeUpscaleBackend:
    name = "fake-upscaler"
    license = MIT
    supported_factors = (2, 4)

    def __init__(self) -> None:
        self.calls = 0

    def variant_for_factor(self, factor: int) -> str:
        return f"fake-{factor}x"

    def upscale(self, image, *, factor: int, tile: int = 0, **params):
        self.calls += 1
        return np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)


def test_upscale_use_case_preserves_alpha(tmp_path):
    path = tmp_path / "sticker.png"
    from splat.adapters.formats.image import write_png

    write_png(path, sample_rgba((2, 2), color=(10, 20, 30, 128)))

    result = UpscaleUseCase(FakeUpscaleBackend()).execute(path, factor=2)

    assert result.image.shape == (4, 4, 4)
    assert result.image[0, 0].tolist() == [10, 20, 30, 128]
    assert result.source_width == 2
    assert result.output_width == 4


def test_upscale_use_case_rejects_unsupported_factor(tmp_path):
    path = write_sample_png(tmp_path / "image.png", (2, 2))

    with pytest.raises(SplatDomainError, match="--factor 2, 4"):
        UpscaleUseCase(FakeUpscaleBackend()).execute(path, factor=3)


def test_run_upscale_uses_cache_key(tmp_path):
    cache = FilesystemManifestRepository(tmp_path / "cache")
    source = cache.put_external(
        write_sample_png(tmp_path / "image.png", (2, 2)),
        kind=ManifestKind.IMAGE,
    )
    backend = FakeUpscaleBackend()

    first = run_upscale(
        backend,
        cache,
        model_name="fake-upscaler",
        input_asset=source,
        params={"factor": 2, "tile": 0},
    )
    second = run_upscale(
        backend,
        cache,
        model_name="fake-upscaler",
        input_asset=source,
        params={"factor": 2, "tile": 0},
    )

    assert first.id == second.id
    assert backend.calls == 1
    assert read_rgb_or_rgba(first.content_path).shape == (4, 4, 3)
    assert first.metadata.variant == "fake-2x"
