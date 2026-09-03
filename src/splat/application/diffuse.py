from pathlib import Path

from splat.ports.diffusion import DiffusionBackend


class DiffuseUseCase:
    def __init__(self, backend: DiffusionBackend) -> None:
        self._backend = backend

    def execute(self, prompt: str, *, output_path: Path, **params) -> Path:
        return self._backend.diffuse(prompt, output_path=output_path, **params)
