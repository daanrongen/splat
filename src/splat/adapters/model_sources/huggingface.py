"""Wraps the HuggingFace Hub cache as a ModelSource — the only adapter that
performs a network call to fetch model weights via the generic `pull()`
port. Raw HF snapshots land in HF_HOME, shared with other Hugging Face tools.
`SPLAT_MODEL_CACHE_DIR` is reserved for app-specific materialized artifacts.
Uses `scan_cache_dir()` rather than a strict full-snapshot check, since a
partial download (e.g. via `allow_patterns`) should still count as cached.

The CoreML/MLX adapters don't use the standard HF blob/snapshot cache layout
at all — they download real files (CoreML can't compile a symlinked
weight.bin) or store a converted copy under
`model_cache_dir()/<kind>/<org>--<name>/`. `_bespoke_local_dir()` covers
that case by name convention alone, since it's the one thing every such
adapter shares.
"""

import shutil
from pathlib import Path

from huggingface_hub import scan_cache_dir, snapshot_download

from splat.paths import model_cache_dir


class HuggingFaceModelSource:
    name = "huggingface"

    def _bespoke_local_dir(self, model_id: str) -> Path | None:
        suffix = model_id.replace("/", "--")
        for candidate in model_cache_dir().glob(f"*/{suffix}"):
            if candidate.is_dir() and any(candidate.iterdir()):
                return candidate
        return None

    def pull(self, model_id: str, *, revision: str | None = None) -> Path:
        path = snapshot_download(repo_id=model_id, revision=revision)
        return Path(path)

    def is_cached(self, model_id: str) -> bool:
        return self.local_path(model_id) is not None

    def local_path(self, model_id: str) -> Path | None:
        try:
            info = scan_cache_dir()
        except Exception:
            info = None
        if info is not None:
            for repo in info.repos:
                if repo.repo_id == model_id and repo.revisions:
                    return Path(next(iter(repo.revisions)).snapshot_path)
        return self._bespoke_local_dir(model_id)

    def remove(self, model_id: str) -> None:
        path = self.local_path(model_id)
        if path is None:
            return
        # snapshot_download resolves to .../models--org--name/snapshots/<rev>.
        if path.parent.name == "snapshots" and path.parent.parent.name.startswith("models--"):
            shutil.rmtree(path.parent.parent, ignore_errors=True)
            return
        shutil.rmtree(path, ignore_errors=True)

    def list_cached(self) -> list[str]:
        names: set[str] = set()
        try:
            info = scan_cache_dir()
        except Exception:
            return []
        names.update(repo.repo_id for repo in info.repos)
        return sorted(names)
