from pathlib import Path

from splat.ports.generation import ImageGenerationBackend


class GenerateUseCase:
    def __init__(self, backend: ImageGenerationBackend) -> None:
        self._backend = backend

    def execute(self, prompt: str, *, output_path: Path, **params) -> Path:
        return self._backend.generate(prompt, output_path=output_path, **params)
