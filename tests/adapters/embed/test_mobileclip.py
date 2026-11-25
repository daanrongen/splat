from pathlib import Path

import pytest

from splat.adapters.embed.mobileclip import MobileCLIPOpenCLIPBackend
from splat.domain.value_objects import APPLE_AMLR

WEIGHTS = Path("/models/timm--MobileCLIP2-S0-OpenCLIP")


class FakeModel:
    def eval(self):
        return self

    def to(self, device):
        return self


@pytest.fixture
def open_clip(mocker):
    module = mocker.patch("open_clip.create_model_and_transforms")
    module.return_value = (FakeModel(), None, object())
    mocker.patch("open_clip.get_tokenizer")
    mocker.patch("timm.utils.reparameterize_model", side_effect=lambda model: model)
    return module


def test_load_reads_the_model_from_the_weights_directory(open_clip):
    backend = MobileCLIPOpenCLIPBackend(weights_paths=(WEIGHTS,), license=APPLE_AMLR, device="cpu")

    backend._load()

    open_clip.assert_called_once_with(f"local-dir:{WEIGHTS}")


def test_load_reads_the_tokenizer_from_the_weights_directory(open_clip, mocker):
    tokenizer = mocker.patch("open_clip.get_tokenizer")
    backend = MobileCLIPOpenCLIPBackend(weights_paths=(WEIGHTS,), license=APPLE_AMLR, device="cpu")

    backend._load()

    tokenizer.assert_called_once_with(f"local-dir:{WEIGHTS}")


def test_load_only_reads_the_weights_once(open_clip):
    backend = MobileCLIPOpenCLIPBackend(weights_paths=(WEIGHTS,), license=APPLE_AMLR, device="cpu")

    backend._load()
    backend._load()

    open_clip.assert_called_once()
