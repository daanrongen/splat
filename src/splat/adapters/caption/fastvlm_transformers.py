from pathlib import Path
from typing import Any

from splat.domain.errors import SplatDomainError
from splat.domain.value_objects import ModelLicense

_IMAGE_TOKEN_INDEX = -200


def _resolve_device(device: str) -> str:
    if device != "auto":
        if device not in ("cpu", "mps"):
            raise SplatDomainError("FastVLM caption device must be 'auto', 'cpu', or 'mps'.")
        return device

    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class FastVLMTransformersBackend:
    name = "fastvlm-transformers"

    def __init__(self, *, hf_repo_id: str, license: ModelLicense, device: str = "auto") -> None:
        self._hf_repo_id = hf_repo_id
        self.license = license
        self._device = _resolve_device(device)
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def _load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._hf_repo_id,
            trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self._hf_repo_id,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to(self._device)
        self._model.eval()

    def _input_ids_for_prompt(self, prompt: str):
        import torch

        rendered = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": f"<image>\n{prompt}"}],
            add_generation_prompt=True,
            tokenize=False,
        )
        pre, post = rendered.split("<image>", 1)
        pre_ids = self._tokenizer(pre, return_tensors="pt", add_special_tokens=False).input_ids
        post_ids = self._tokenizer(post, return_tensors="pt", add_special_tokens=False).input_ids
        image_token = torch.tensor([[_IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
        return torch.cat([pre_ids, image_token, post_ids], dim=1).to(self._model.device)

    def _pixel_values(self, image_path: Path):
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        pixel_values = self._model.get_vision_tower().image_processor(
            images=image, return_tensors="pt"
        )["pixel_values"]
        return pixel_values.to(self._model.device, dtype=self._model.dtype)

    def caption(
        self,
        image_path: Path,
        *,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **params,
    ) -> str:
        self._load()

        import torch

        input_ids = self._input_ids_for_prompt(prompt)
        attention_mask = torch.ones_like(input_ids, device=self._model.device)
        generate_kwargs = {
            "inputs": input_ids,
            "attention_mask": attention_mask,
            "images": self._pixel_values(image_path),
            "max_new_tokens": max_tokens,
        }
        if temperature > 0.0:
            generate_kwargs.update({"do_sample": True, "temperature": temperature})
        else:
            generate_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self._model.generate(**generate_kwargs)

        generated_ids = output_ids[0, input_ids.shape[1] :]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        if text:
            return text
        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
