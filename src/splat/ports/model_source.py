from pathlib import Path
from typing import Protocol


class ModelSource(Protocol):
    """Pulls/caches model weights on demand. Weights are never bundled with
    the package — this is the only place a network call to fetch weights
    happens."""

    def pull(self, model_id: str, *, revision: str | None = None) -> Path: ...

    def is_cached(self, model_id: str) -> bool: ...

    def local_path(self, model_id: str) -> Path | None: ...

    def remove(self, model_id: str) -> None: ...

    def list_cached(self) -> list[str]: ...
