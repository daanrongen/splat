"""Fakes a slow download so the `models pull` progress bar can be tried without network."""

import time
from collections.abc import Callable
from pathlib import Path

from splat.adapters.model_sources.huggingface import HuggingFaceModelSource
from splat.paths import model_cache_dir

_TOTAL = 64 * 1024 * 1024
_STEP = 1024 * 1024
_DELAY = 0.05


class StubModelSource(HuggingFaceModelSource):
    name = "stub"

    def pull(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> Path:
        cached = self.local_path(model_id)
        if cached is not None:
            return cached
        for done in range(_STEP, _TOTAL + 1, _STEP):
            time.sleep(_DELAY)
            if on_progress is not None:
                on_progress(done, _TOTAL)
        directory = model_cache_dir() / "stub" / model_id.replace("/", "--")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "stub.txt").write_text("stub")
        return directory
