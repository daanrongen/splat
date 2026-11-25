from pathlib import Path

import coremltools as ct
import numpy as np
from PIL import Image

from splat.adapters.depth.coreml_depth_anything_v2 import CoreMLDepthAnythingV2Backend
from splat.domain.value_objects import APACHE_2_0
from tests.image_helpers import write_sample_png

WEIGHTS = Path("models/apple--coreml-depth-anything-v2-small")


class FakeMLModel:
    def __init__(self, path, compute_units):
        self.path = path
        self.compute_units = compute_units
        self.predict_calls = []

    def predict(self, inputs):
        self.predict_calls.append(inputs)
        image = inputs["image"]
        depth = np.linspace(0.0, 1.0, image.width * image.height, dtype=np.float32).reshape(
            image.height, image.width
        )
        return {"depth": Image.fromarray(depth, mode="F")}


def _patch(mocker, models: list):
    def fake_ml_model(path, compute_units):
        model = FakeMLModel(path, compute_units)
        models.append(model)
        return model

    mocker.patch(
        "splat.adapters.depth.coreml_depth_anything_v2.ct.models.MLModel",
        side_effect=fake_ml_model,
    )


def test_estimate_returns_relative_depth_map(mocker, tmp_path):
    models: list = []
    _patch(mocker, models)

    image_path = tmp_path / "scene.png"
    write_sample_png(image_path, (10, 8))

    backend = CoreMLDepthAnythingV2Backend(
        weights_paths=(WEIGHTS,), license=APACHE_2_0, device="cpu"
    )
    result = backend.estimate(image_path)

    assert result.depth.shape == (8, 10)
    assert result.depth.dtype == np.float32
    assert result.focal_length_px is None
    assert result.field_of_view_deg is None
    assert result.units == "disparity"
    assert result.metadata["source_model"] == "apple/coreml-depth-anything-v2-small"
    assert models[0].compute_units == ct.ComputeUnit.CPU_ONLY


def test_load_only_loads_weights_once(mocker, tmp_path):
    models: list = []
    _patch(mocker, models)

    image_path = tmp_path / "scene.png"
    write_sample_png(image_path, (10, 8))

    backend = CoreMLDepthAnythingV2Backend(
        weights_paths=(WEIGHTS,), license=APACHE_2_0, device="cpu"
    )
    backend.estimate(image_path)
    backend.estimate(image_path)

    assert len(models) == 1


def test_init_stores_fields():
    backend = CoreMLDepthAnythingV2Backend(
        weights_paths=(WEIGHTS,), license=APACHE_2_0, device="cpu"
    )
    assert backend.name == "depth-anything-v2-coreml"
    assert backend.license is APACHE_2_0


def test_load_uses_the_f16_package_from_the_weights_path(mocker, tmp_path):
    models: list = []
    _patch(mocker, models)
    image_path = tmp_path / "scene.png"
    write_sample_png(image_path, (10, 8))

    CoreMLDepthAnythingV2Backend(weights_paths=(WEIGHTS,), license=APACHE_2_0).estimate(image_path)

    assert models[0].path == str(WEIGHTS / "DepthAnythingV2SmallF16.mlpackage")
