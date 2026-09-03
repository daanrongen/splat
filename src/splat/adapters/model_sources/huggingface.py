"""Wraps the HuggingFace Hub cache as a ModelSource — the only adapter that
performs a network call to fetch model weights via the generic `pull()`
port. Weights land under SPLAT_MODEL_CACHE_DIR (see splat.paths); several
other adapters use HF's own default cache (HF_HOME) or a bespoke local_dir
directly, so `is_cached()`/`local_path()` check both roots. Uses
`scan_cache_dir()` rather than a strict full-snapshot check, since a
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

    def _cache_dirs(self) -> list[str | None]:
        return [str(model_cache_dir()), None]  # None = huggingface_hub's own default (HF_HOME)

    def _bespoke_local_dir(self, model_id: str) -> Path | None:
        suffix = model_id.replace("/", "--")
        for candidate in model_cache_dir().glob(f"*/{suffix}"):
            if candidate.is_dir() and any(candidate.iterdir()):
                return candidate
        return None

    def pull(self, model_id: str, *, revision: str | None = None) -> Path:
        path = snapshot_download(
            repo_id=model_id, revision=revision, cache_dir=str(model_cache_dir())
        )
        return Path(path)

    def is_cached(self, model_id: str) -> bool:
        return self.local_path(model_id) is not None

    def local_path(self, model_id: str) -> Path | None:
        for cache_dir in self._cache_dirs():
            try:
                info = scan_cache_dir(cache_dir=cache_dir) if cache_dir else scan_cache_dir()
            except Exception:
                continue
            for repo in info.repos:
                if repo.repo_id == model_id and repo.revisions:
                    return Path(next(iter(repo.revisions)).snapshot_path)
        return self._bespoke_local_dir(model_id)

    def remove(self, model_id: str) -> None:
        path = self.local_path(model_id)
        if path is None:
            return
        # snapshot_download resolves to .../models--org--name/snapshots/<rev>
        model_root = path.parent.parent
        shutil.rmtree(model_root, ignore_errors=True)

    def list_cached(self) -> list[str]:
        names: set[str] = set()
        for cache_dir in self._cache_dirs():
            try:
                info = scan_cache_dir(cache_dir=cache_dir) if cache_dir else scan_cache_dir()
            except Exception:
                continue
            names.update(repo.repo_id for repo in info.repos)
        return sorted(names)
