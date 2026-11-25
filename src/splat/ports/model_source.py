from collections.abc import Callable
from pathlib import Path
from typing import Protocol


class ModelSource(Protocol):
    """Pulls/caches model weights on demand into one directory per repo. Weights are
    never bundled with the package; this is the only place a network call to fetch
    weights happens. `files` are the glob patterns a model needs, empty meaning all."""

    def pull(
        self,
        model_id: str,
        *,
        files: tuple[str, ...] = (),
        on_progress: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Reports `(bytes_done, bytes_total)` as the download advances."""
        ...

    def is_cached(self, model_id: str, files: tuple[str, ...] = ()) -> bool: ...

    def local_path(self, model_id: str, files: tuple[str, ...] = ()) -> Path | None: ...

    def remove(self, model_id: str) -> None: ...

    def list_cached(self) -> list[str]: ...

    def size_on_disk(self, model_id: str) -> int:
        """Bytes this model occupies locally, 0 if it isn't cached."""
        ...

    def list_legacy(self) -> dict[str, int]:
        """Repos an older layout left outside the store, with their size in bytes."""
        ...

    def remove_legacy(self, model_id: str) -> None: ...
