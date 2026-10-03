import sys
import types

import torch

from splat.adapters.caption.fastvlm_transformers import (
    FastVLMTransformersBackend,
    _clean_caption,
)
from splat.domain.value_objects import APPLE_AMLR
from tests.image_helpers import write_sample_png


class FakeTokenizer:
    eos_token_id = 151645

    @classmethod
    def from_pretrained(cls, repo_id, *, trust_remote_code):
        return cls(repo_id, trust_remote_code)

    def __init__(self, repo_id, trust_remote_code) -> None:
        self.repo_id = repo_id
        self.trust_remote_code = trust_remote_code

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        assert add_generation_prompt is True
        assert tokenize is False
        return f"before {messages[0]['content']} after"

    def __call__(self, text, *, return_tensors, add_special_tokens):
        assert return_tensors == "pt"
        assert add_special_tokens is False
        token = 1 if "before" in text else 2
        return types.SimpleNamespace(input_ids=torch.tensor([[token]], dtype=torch.long))

    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens is True
        return " A small scene. "


class FakeVisionTower:
    def image_processor(self, *, images, return_tensors):
        assert return_tensors == "pt"
        assert images.mode == "RGB"
        return {"pixel_values": torch.ones((1, 3, 2, 2), dtype=torch.float32)}


class FakeModel:
    device = torch.device("cpu")
    dtype = torch.float32

    @classmethod
    def from_pretrained(cls, repo_id, *, torch_dtype, trust_remote_code):
        assert repo_id == "apple/FastVLM-0.5B"
        assert torch_dtype == torch.float32
        assert trust_remote_code is True
        return cls()

    def to(self, device):
        self.device = torch.device(device)
        return self

    def eval(self):
        return None

    def get_vision_tower(self):
        return FakeVisionTower()

    def generate(self, **kwargs):
        assert kwargs["max_new_tokens"] == 12
        assert kwargs["do_sample"] is False
        assert kwargs["eos_token_id"] == FakeTokenizer.eos_token_id  # stop at the chat-end token
        assert kwargs["images"].shape == (1, 3, 2, 2)
        input_ids = kwargs["inputs"]
        return torch.cat([input_ids, torch.tensor([[10, 11]], dtype=input_ids.dtype)], dim=1)


def test_fastvlm_transformers_backend_generates_caption(monkeypatch, tmp_path):
    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = FakeModel
    transformers.AutoTokenizer = FakeTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    image_path = write_sample_png(tmp_path / "scene.png", (2, 2))
    backend = FastVLMTransformersBackend(
        hf_repo_id="apple/FastVLM-0.5B", license=APPLE_AMLR, device="cpu"
    )

    result = backend.caption(
        image_path,
        prompt="Describe",
        max_tokens=12,
        temperature=0.0,
    )

    assert result == "A small scene."


def test_clean_caption_drops_chat_template_scaffolding():
    raw = "A red robot toy.\n<end of detailed answer>\nAnswer: A red robot toy.\nAnswer: A"

    assert _clean_caption(raw) == "A red robot toy."


def test_clean_caption_strips_leading_answer_prefix():
    assert _clean_caption("Answer: A fox.") == "A fox."
