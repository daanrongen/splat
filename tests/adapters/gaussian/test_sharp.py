import numpy as np
import pytest
from PIL import Image

from splat.adapters.gaussian.sharp import SharpBackend, _focal_px, _resolve_device
from splat.domain.errors import ReconstructionBackendError
from splat.domain.value_objects import APPLE_AMLR
from splat.registry.gaussian import GAUSSIAN_CATALOG


def _backend(**kwargs) -> SharpBackend:
    return SharpBackend(license=APPLE_AMLR, **kwargs)


def test_accepts_exactly_one_image():
    """The whole point: this is the only backend a single diffused image can
    reach, so the contract has to say (1, 1) without loading weights."""
    assert SharpBackend.required_image_count() == (1, 1)


def test_declares_metric_scale():
    assert SharpBackend.provides_metric_scale is True


def test_reconstruct_rejects_multiple_images(tmp_path):
    with pytest.raises(ReconstructionBackendError, match="exactly one image, got 2"):
        _backend(weights_path=tmp_path).reconstruct([tmp_path / "a.png", tmp_path / "b.png"])


def test_checkpoint_error_names_the_pull_command(tmp_path):
    with pytest.raises(ReconstructionBackendError, match="models pull sharp"):
        _backend(weights_path=tmp_path)._checkpoint()


def test_checkpoint_error_when_weights_were_never_resolved():
    with pytest.raises(ReconstructionBackendError, match="models pull sharp"):
        _backend()._checkpoint()


def test_checkpoint_found_in_the_snapshot_dir(tmp_path):
    (tmp_path / "sharp_2572gikvuh.pt").write_bytes(b"")
    (tmp_path / "README.md").write_text("not a checkpoint")

    assert _backend(weights_path=tmp_path)._checkpoint().name == "sharp_2572gikvuh.pt"


def test_resolve_device_rejects_unknown_values():
    with pytest.raises(ReconstructionBackendError, match="must be 'auto'"):
        _resolve_device("tpu")


def test_resolve_device_passes_explicit_values_through():
    assert _resolve_device("cpu") == "cpu"


def test_focal_px_matches_the_reference_35mm_conversion(tmp_path):
    """Apple's `convert_focallength`: f_mm * diag_px / diag_35mm."""
    expected = 30.0 * np.sqrt(512**2 + 512**2) / np.sqrt(36**2 + 24**2)

    assert _focal_px(tmp_path / "x.png", 512, 512, 30.0) == pytest.approx(expected)


def _jpeg(path, focal_35mm=None):
    exif = Image.Exif()
    if focal_35mm is not None:
        exif.get_ifd(0x8769)[0xA405] = focal_35mm
    Image.new("RGB", (8, 8)).save(path, exif=exif.tobytes())
    return path


def test_focal_px_prefers_the_exif_35mm_focal_length(tmp_path):
    photo = _jpeg(tmp_path / "phone.jpg", focal_35mm=24)

    assert _focal_px(photo, 512, 512, 30.0) == pytest.approx(
        _focal_px(tmp_path / "x.png", 512, 512, 24.0)
    )


def test_focal_px_falls_back_when_exif_has_no_focal_length(tmp_path):
    photo = _jpeg(tmp_path / "scan.jpg")

    assert _focal_px(photo, 512, 512, 30.0) == pytest.approx(
        _focal_px(tmp_path / "x.png", 512, 512, 30.0)
    )


def test_catalog_entry_is_non_commercial_and_single_image():
    descriptor = GAUSSIAN_CATALOG["sharp"]

    assert descriptor.hf_repo_id == "apple/Sharp"
    assert descriptor.license.spdx_id == "Apple-ML-Research"
    assert descriptor.license.is_commercial is False
    assert descriptor.runtime == "torch"
    assert descriptor.backend_cls.required_image_count() == (1, 1)
