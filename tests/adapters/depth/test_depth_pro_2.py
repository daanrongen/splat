import math

import numpy as np
import torch

from splat.adapters.depth.depth_pro_2 import DepthPro2Backend, _resolve_device
from splat.domain.value_objects import APPLE_AMLR
from tests.image_helpers import write_sample_png

H = "apple/DepthPro2-H"
FOCAL = "apple/DepthPro2-focal"


class FakeFov:
    def __init__(self, precision):
        self.precision = precision
        self.state = None

    def load_state_dict(self, state):
        self.state = state


class FakeModel:
    def __init__(self, depth, focal):
        self.fov = None
        self.to_device = None
        self.eval_called = False
        self.infer_calls = 0
        self._prediction = {"depth": depth, "focallength_px": torch.tensor(focal)}

    def to(self, device):
        self.to_device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def infer(self, x):
        self.infer_calls += 1
        return self._prediction


def _patch_runtime(mocker, model):
    from_pretrained = mocker.patch("depth_pro_2.DepthPro2.from_pretrained", return_value=model)
    mocker.patch("depth_pro_2.network.fov.FOVEstimator", FakeFov)
    download = mocker.patch("huggingface_hub.hf_hub_download", return_value="focal.safetensors")
    load_file = mocker.patch("safetensors.torch.load_file", return_value={"w": 1})
    return from_pretrained, download, load_file


def _backend():
    return DepthPro2Backend(hf_repo_id=H, focal_repo_id=FOCAL, license=APPLE_AMLR, device="cpu")


def _image(tmp_path):
    path = tmp_path / "scene.png"
    write_sample_png(path, (10, 8))
    return path


def test_explicit_device_is_kept():
    assert _resolve_device("cpu") == "cpu"


def test_auto_device_prefers_mps(mocker):
    mocker.patch("torch.backends.mps.is_available", return_value=True)

    assert _resolve_device("auto") == "mps"


def test_auto_device_falls_back_to_cpu(mocker):
    mocker.patch("torch.backends.mps.is_available", return_value=False)

    assert _resolve_device("auto") == "cpu"


def test_load_pulls_both_repos_once(mocker, tmp_path):
    model = FakeModel(torch.zeros((8, 10)), 10.0)
    from_pretrained, download, load_file = _patch_runtime(mocker, model)
    backend = _backend()

    backend.estimate(_image(tmp_path))
    backend.estimate(_image(tmp_path))

    from_pretrained.assert_called_once_with(H, precision=torch.float32)
    download.assert_called_once_with(FOCAL, "model.safetensors")
    load_file.assert_called_once_with("focal.safetensors")
    assert model.fov.state == {"w": 1}
    assert model.to_device == "cpu"
    assert model.eval_called is True


def test_estimate_returns_metric_depth_map(mocker, tmp_path):
    depth = torch.arange(80, dtype=torch.float32).reshape(8, 10)
    _patch_runtime(mocker, FakeModel(depth, 10.0))

    result = _backend().estimate(_image(tmp_path))

    assert result.depth.dtype == np.float32
    np.testing.assert_array_equal(result.depth, depth.numpy())
    assert result.units == "metres"


def test_estimate_derives_field_of_view_from_focal_length(mocker, tmp_path):
    _patch_runtime(mocker, FakeModel(torch.zeros((8, 10)), 5.0))

    result = _backend().estimate(_image(tmp_path))

    assert result.focal_length_px == 5.0
    assert math.isclose(result.field_of_view_deg, 90.0)


def test_estimate_records_effective_params(mocker, tmp_path):
    _patch_runtime(mocker, FakeModel(torch.zeros((8, 10)), 10.0))

    result = _backend().estimate(_image(tmp_path))

    assert result.metadata == {"source_model": H, "device": "cpu", "precision": "float32"}
