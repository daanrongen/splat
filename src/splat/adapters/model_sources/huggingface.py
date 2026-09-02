"""Wraps the HuggingFace Hub cache as a ModelSource — the only adapter that
performs a network call to fetch model weights. Weights land in HF's own
cache dir (HF_HOME / SPLAT_MODEL_CACHE_DIR, set via mise.toml) rather than
package data, and are never bundled."""

import os
import shutil
from pathlib import Path

from huggingface_hub import scan_cache_dir, snapshot_download


class HuggingFaceModelSource:
    name = "huggingface"

    def _cache_dir(self) -> str | None:
        return os.environ.get("SPLAT_MODEL_CACHE_DIR")

    def pull(self, model_id: str, *, revision: str | None = None) -> Path:
        path = snapshot_download(repo_id=model_id, revision=revision, cache_dir=self._cache_dir())
        return Path(path)

    def is_cached(self, model_id: str) -> bool:
        return self.local_path(model_id) is not None

    def local_path(self, model_id: str) -> Path | None:
        try:
            path = snapshot_download(
                repo_id=model_id, cache_dir=self._cache_dir(), local_files_only=True
            )
        except Exception:
            return None
        return Path(path)

    def remove(self, model_id: str) -> None:
        path = self.local_path(model_id)
        if path is None:
            return
        # snapshot_download resolves to .../models--org--name/snapshots/<rev>
        model_root = path.parent.parent
        shutil.rmtree(model_root, ignore_errors=True)

    def list_cached(self) -> list[str]:
        cache_dir = self._cache_dir()
        info = scan_cache_dir(cache_dir=cache_dir) if cache_dir else scan_cache_dir()
        return [repo.repo_id for repo in info.repos]
