import numpy as np
import pytest
from PIL import Image

from splat.adapters.segment.coreml_sam2 import CoreMLSam2Backend
from splat.domain.errors import SplatDomainError
from splat.domain.value_objects import APACHE_2_0
from tests.image_helpers import write_sample_png


class FakeImageEncoder:
    def __init__(self):
        self.inputs = []

    def predict(self, inputs):
        self.inputs.append(inputs)
        zeros = np.zeros((1, 1), dtype=np.float32)
        return {"image_embedding": zeros, "feats_s0": zeros, "feats_s1": zeros}


class FakePromptEncoder:
    def predict(self, inputs):
        zeros = np.zeros((1, 1), dtype=np.float32)
        return {"sparse_embeddings": zeros, "dense_embeddings": zeros}


class FakeMaskDecoder:
    def predict(self, inputs):
        return {
            "scores": np.array([[0.5]], dtype=np.float32),
            "low_res_masks": np.zeros((1, 1, 4, 4), dtype=np.float32),
        }


def test_segment_passes_pil_image_to_image_encoder(tmp_path):
    backend = CoreMLSam2Backend(hf_repo_id="apple/coreml-sam2.1-tiny", license=APACHE_2_0)
    encoder = FakeImageEncoder()
    backend._image_encoder = encoder
    backend._prompt_encoder = FakePromptEncoder()
    backend._mask_decoder = FakeMaskDecoder()
    image_path = write_sample_png(tmp_path / "in.png")

    backend.segment(image_path, points_per_side=1)

    assert isinstance(encoder.inputs[0]["image"], Image.Image)
    assert encoder.inputs[0]["image"].size == (1024, 1024)


class RecordingPromptEncoder(FakePromptEncoder):
    def __init__(self):
        self.inputs = []

    def predict(self, inputs):
        self.inputs.append(inputs)
        return super().predict(inputs)


class FullMaskDecoder:
    def predict(self, inputs):
        return {
            "scores": np.array([[0.9]], dtype=np.float32),
            "low_res_masks": np.ones((1, 1, 4, 4), dtype=np.float32),
        }


def _prompted_backend():
    backend = CoreMLSam2Backend(hf_repo_id="apple/coreml-sam2.1-tiny", license=APACHE_2_0)
    backend._image_encoder = FakeImageEncoder()
    backend._prompt_encoder = RecordingPromptEncoder()
    backend._mask_decoder = FullMaskDecoder()
    return backend


def test_box_prompt_is_two_corner_points_scaled_to_the_encoder_input(tmp_path):
    backend = _prompted_backend()
    image_path = write_sample_png(tmp_path / "in.png", (512, 256))

    (sticker,) = backend.segment(image_path, box=[0, 0, 512, 128])

    sent = backend._prompt_encoder.inputs[0]
    assert sent["labels"].tolist() == [[2, 3]]
    assert sent["points"].tolist() == [[[0.0, 0.0], [1024.0, 512.0]]]
    assert sticker.bbox == (0, 0, 512, 256)


def test_single_point_is_doubled_and_keeps_its_label(tmp_path):
    backend = _prompted_backend()

    backend.segment(write_sample_png(tmp_path / "in.png", (512, 512)), points=[[256, 256, 0]])

    sent = backend._prompt_encoder.inputs[0]
    assert sent["labels"].tolist() == [[0, 0]]
    assert sent["points"].tolist() == [[[512.0, 512.0], [512.0, 512.0]]]


def test_too_many_points_or_box_with_points_are_rejected(tmp_path):
    backend = _prompted_backend()
    image_path = write_sample_png(tmp_path / "in.png", (64, 64))

    with pytest.raises(SplatDomainError):
        backend.segment(image_path, points=[[1, 1, 1], [2, 2, 1], [3, 3, 1]])
    with pytest.raises(SplatDomainError):
        backend.segment(image_path, points=[[1, 1, 1]], box=[0, 0, 5, 5])
