from pathlib import Path

from splat.ports.caption import CaptioningBackend


class CaptionUseCase:
    def __init__(self, backend: CaptioningBackend) -> None:
        self._backend = backend

    def execute(
        self,
        image_path: Path,
        *,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **params,
    ) -> str:
        return self._backend.caption(
            image_path,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **params,
        )
