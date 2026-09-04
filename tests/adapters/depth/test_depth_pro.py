import numpy as np
import torch

from splat.adapters.depth.depth_pro import DepthProBackend, _resolve_device
from splat.domain.value_objects import APPLE_ASCL
from tests.image_helpers import write_sample_png


class FakeBatchFeature(dict):
    def to(self, device):
        return self


class FakeProcessor:
    def __init__(self, post_process_result):
        self.post_process_result = post_process_result
        self.call_count = 0

    def __call__(self, images, return_tensors):
        return FakeBatchFeature()

    def post_process_depth_estimation(self, outputs, target_sizes):
        self.call_count += 1
        return self.post_process_result


class FakeModel:
    def __init__(self):
        self.to_device = None
        self.eval_called = False

    def to(self, device):
        self.to_device = device
        return self

    def eval(self):
        self.eval_called = True

    def __call__(self, **kwargs):
        return "raw_outputs"


def _patch_from_pretrained(mocker, processor: FakeProcessor, model: FakeModel):
    processor_ctor = mocker.patch(
        "splat.adapters.depth.depth_pro.DepthProImageProcessor.from_pretrained",
        return_value=processor,
    )
    model_ctor = mocker.patch(
        "splat.adapters.depth.depth_pro.DepthProForDepthEstimation.from_pretrained",
        return_value=model,
    )
    return processor_ctor, model_ctor


def test_resolve_device_auto_picks_mps_when_available(mocker):
    mocker.patch(
        "splat.adapters.depth.depth_pro.torch.backends.mps.is_available", return_value=True
    )
    assert _resolve_device("auto") == "mps"


def test_resolve_device_auto_falls_back_to_cpu(mocker):
    mocker.patch(
        "splat.adapters.depth.depth_pro.torch.backends.mps.is_available", return_value=False
    )
    assert _resolve_device("auto") == "cpu"


def test_resolve_device_explicit_passthrough(mocker):
    mocker.patch(
        "splat.adapters.depth.depth_pro.torch.backends.mps.is_available", return_value=True
    )
    assert _resolve_device("cpu") == "cpu"


def test_init_resolves_device_and_stores_fields():
    backend = DepthProBackend(hf_repo_id="apple/DepthPro-hf", license=APPLE_ASCL, device="cpu")
    assert backend._hf_repo_id == "apple/DepthPro-hf"
    assert backend.license is APPLE_ASCL
    assert backend._device == "cpu"
    assert backend.name == "depth-pro"


def test_load_only_downloads_weights_once(mocker, tmp_path):
    depth = torch.zeros((8, 10))
    processor = FakeProcessor(
        [{"predicted_depth": depth, "focal_length": None, "field_of_view": None}]
    )
    model = FakeModel()
    processor_ctor, model_ctor = _patch_from_pretrained(mocker, processor, model)

    image_path = tmp_path / "scene.png"
    write_sample_png(image_path, (10, 8))

    backend = DepthProBackend(hf_repo_id="apple/DepthPro-hf", license=APPLE_ASCL, device="cpu")
    backend.estimate(image_path)
    backend.estimate(image_path)

    processor_ctor.assert_called_once_with("apple/DepthPro-hf")
    model_ctor.assert_called_once_with("apple/DepthPro-hf")
    assert model.to_device == "cpu"
    assert model.eval_called is True


def test_estimate_returns_populated_depth_map(mocker, tmp_path):
    depth = torch.arange(80, dtype=torch.float32).reshape(8, 10)
    processor = FakeProcessor(
        [{"predicted_depth": depth, "focal_length": 1520.0, "field_of_view": 19.125}]
    )
    model = FakeModel()
    _patch_from_pretrained(mocker, processor, model)

    image_path = tmp_path / "scene.png"
    write_sample_png(image_path, (10, 8))

    backend = DepthProBackend(hf_repo_id="apple/DepthPro-hf", license=APPLE_ASCL, device="cpu")
    result = backend.estimate(image_path)

    assert result.depth.shape == (8, 10)
    assert result.depth.dtype == np.float32
    np.testing.assert_array_equal(result.depth, depth.numpy())
    assert result.focal_length_px == 1520.0
    assert result.field_of_view_deg == 19.125
    assert result.metadata == {"source_model": "apple/DepthPro-hf", "device": "cpu"}
    assert processor.call_count == 1


def test_estimate_handles_missing_focal_length_and_fov(mocker, tmp_path):
    depth = torch.zeros((8, 10))
    processor = FakeProcessor([{"predicted_depth": depth}])
    model = FakeModel()
    _patch_from_pretrained(mocker, processor, model)

    image_path = tmp_path / "scene.png"
    write_sample_png(image_path, (10, 8))

    backend = DepthProBackend(hf_repo_id="apple/DepthPro-hf", license=APPLE_ASCL, device="cpu")
    result = backend.estimate(image_path)

    assert result.focal_length_px is None
    assert result.field_of_view_deg is None
