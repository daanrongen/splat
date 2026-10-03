import numpy as np
from PIL import Image

from splat.adapters.segment.coreml_sam2 import CoreMLSam2Backend
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
