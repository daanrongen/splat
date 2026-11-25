import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from splat.adapters.diffuse.coreml_stable_diffusion import (
    _SUBFOLDER,
    _TOKENIZER_SUBFOLDER,
    CoreMLStableDiffusionBackend,
)
from splat.domain.errors import SplatDomainError
from splat.domain.value_objects import APPLE_ASCL

_PREFIX = "Stable_Diffusion_version_runwayml_stable-diffusion-v1-5"
_COMPONENTS = ["text_encoder", "unet", "vae_decoder", "vae_encoder"]


class FakeMLModel:
    calls: ClassVar[list[tuple[str, dict]]] = []
    paths: ClassVar[list[str]] = []

    def __init__(self, path, compute_units=None) -> None:
        self.path = path
        FakeMLModel.paths.append(path)

    def predict(self, inputs):
        name = next(c for c in _COMPONENTS if self.path.endswith(f"_{c}.mlpackage"))
        FakeMLModel.calls.append((name, inputs))
        if name == "text_encoder":
            return {"last_hidden_state": np.zeros((1, 77, 4), dtype=np.float32)}
        if name == "unet":
            return {"noise_pred": np.zeros((2, 4, 64, 64), dtype=np.float32)}
        if name == "vae_encoder":
            return {"latent": np.ones((1, 8, 64, 64), dtype=np.float32)}
        return {"image": np.zeros((1, 3, 512, 512), dtype=np.float32)}


def _package_dir(local_dir, component):
    return local_dir / _SUBFOLDER / f"{_PREFIX}_{component}.mlpackage"


def _write_tokenizer_files(local_dir):
    tokenizer_dir = local_dir / _TOKENIZER_SUBFOLDER
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    (tokenizer_dir / "vocab.json").write_text(
        json.dumps({"a</w>": 0, "b</w>": 1, "<|startoftext|>": 2, "<|endoftext|>": 3})
    )
    (tokenizer_dir / "merges.txt").write_text("#version: 0.2\na b\n")


def _write_packages(local_dir):
    for component in _COMPONENTS:
        _package_dir(local_dir, component).mkdir(parents=True, exist_ok=True)


def _make_backend(mocker, tmp_path):
    mocker.patch("splat.adapters.diffuse.coreml_stable_diffusion.ct.models.MLModel", FakeMLModel)
    FakeMLModel.calls = []
    FakeMLModel.paths = []
    backend = CoreMLStableDiffusionBackend(
        weights_paths=(tmp_path,),
        package_prefix=_PREFIX,
        sdxl=False,
        license=APPLE_ASCL,
    )
    _write_tokenizer_files(tmp_path)
    _write_packages(tmp_path)
    return backend


def test_load_opens_the_four_packages_from_the_weights_path(mocker, tmp_path):
    backend = _make_backend(mocker, tmp_path)

    backend._load()

    assert [Path(m).name for m in FakeMLModel.paths] == [
        f"{_PREFIX}_{c}.mlpackage" for c in _COMPONENTS
    ]


def test_load_only_opens_the_packages_once(mocker, tmp_path):
    backend = _make_backend(mocker, tmp_path)

    backend._load()
    backend._load()

    assert len(FakeMLModel.paths) == 4


def _names():
    return [name for name, _ in FakeMLModel.calls]


def test_text_to_image_skips_the_vae_encoder(mocker, tmp_path):
    backend = _make_backend(mocker, tmp_path)

    backend.diffuse("a", output_path=tmp_path / "out.png", steps=1)

    assert "vae_encoder" not in _names()


def test_image_to_image_encodes_the_input_image(mocker, tmp_path):
    backend = _make_backend(mocker, tmp_path)
    image = np.full((8, 8, 3), 255, dtype=np.uint8)

    backend.diffuse("a", output_path=tmp_path / "out.png", image=image, steps=2)

    assert _names().count("vae_encoder") == 1


def test_image_to_image_scales_the_input_to_512(mocker, tmp_path):
    backend = _make_backend(mocker, tmp_path)
    image = np.full((8, 6, 3), 255, dtype=np.uint8)

    backend.diffuse("a", output_path=tmp_path / "out.png", image=image, steps=1)

    sample = next(inputs for name, inputs in FakeMLModel.calls if name == "vae_encoder")["z"]
    assert sample.shape == (1, 3, 512, 512)


def test_lower_strength_runs_fewer_denoising_steps(mocker, tmp_path):
    backend = _make_backend(mocker, tmp_path)
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    backend.diffuse("a", output_path=tmp_path / "out.png", image=image, steps=10, strength=0.3)

    assert _names().count("unet") == 3


def test_diffuse_rejects_non_default_resolution(mocker, tmp_path):
    backend = _make_backend(mocker, tmp_path)

    with pytest.raises(SplatDomainError, match="fixed 512x512"):
        backend.diffuse("a", output_path=tmp_path / "out.png", width=768)
