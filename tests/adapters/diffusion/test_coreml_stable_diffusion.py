import json

from splat.adapters.diffusion.coreml_stable_diffusion import (
    _PACKAGE_PREFIX,
    _SUBFOLDER,
    _TOKENIZER_SUBFOLDER,
    CoreMLStableDiffusionBackend,
)
from splat.domain.value_objects import APPLE_ASCL

_COMPONENTS = ["text_encoder", "unet", "vae_decoder"]


class FakeMLModel:
    def __init__(self, path, compute_units=None) -> None:
        self.path = path


def _package_dir(local_dir, component):
    return local_dir / _SUBFOLDER / f"{_PACKAGE_PREFIX}_{component}.mlpackage"


def _write_tokenizer_files(local_dir):
    tokenizer_dir = local_dir / _TOKENIZER_SUBFOLDER
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    (tokenizer_dir / "vocab.json").write_text(json.dumps({"a</w>": 0, "b</w>": 1}))
    (tokenizer_dir / "merges.txt").write_text("#version: 0.2\na b\n")


def _write_packages(local_dir):
    for component in _COMPONENTS:
        _package_dir(local_dir, component).mkdir(parents=True, exist_ok=True)


def _make_backend(mocker, tmp_path):
    mocker.patch(
        "splat.adapters.diffusion.coreml_stable_diffusion.model_cache_dir",
        return_value=tmp_path,
    )
    mocker.patch("splat.adapters.diffusion.coreml_stable_diffusion.ct.models.MLModel", FakeMLModel)
    return CoreMLStableDiffusionBackend(
        hf_repo_id="apple/coreml-stable-diffusion-2-1-base", sdxl=False, license=APPLE_ASCL
    )


def test_load_skips_download_when_files_present(mocker, tmp_path):
    backend = _make_backend(mocker, tmp_path)
    local_dir = backend._local_dir()
    _write_tokenizer_files(local_dir)
    _write_packages(local_dir)
    mock_download = mocker.patch(
        "splat.adapters.diffusion.coreml_stable_diffusion.snapshot_download"
    )

    backend._load()

    mock_download.assert_not_called()
    assert backend._unet is not None


def test_load_downloads_when_files_missing(mocker, tmp_path):
    backend = _make_backend(mocker, tmp_path)
    local_dir = backend._local_dir()

    def _fake_download(*args, **kwargs):
        _write_tokenizer_files(local_dir)
        _write_packages(local_dir)

    mock_download = mocker.patch(
        "splat.adapters.diffusion.coreml_stable_diffusion.snapshot_download",
        side_effect=_fake_download,
    )

    backend._load()

    mock_download.assert_called_once()
    assert backend._unet is not None


def test_load_only_downloads_once_per_process(mocker, tmp_path):
    backend = _make_backend(mocker, tmp_path)
    local_dir = backend._local_dir()

    def _fake_download(*args, **kwargs):
        _write_tokenizer_files(local_dir)
        _write_packages(local_dir)

    mock_download = mocker.patch(
        "splat.adapters.diffusion.coreml_stable_diffusion.snapshot_download",
        side_effect=_fake_download,
    )

    backend._load()
    backend._load()

    mock_download.assert_called_once()
