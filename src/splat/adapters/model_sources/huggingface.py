"""Wraps the HuggingFace Hub cache as a ModelSource — the only adapter that
performs a network call to fetch model weights via the generic `pull()`
port. Raw HF snapshots land in HF_HOME, shared with other Hugging Face tools.
`SPLAT_MODEL_CACHE_DIR` is reserved for app-specific materialized artifacts.
Reads the hub cache layout directly (`models--org--name/snapshots/<rev>`)
rather than importing `huggingface_hub`, so listing stays light; a partial
download (e.g. via `allow_patterns`) still counts as cached.

The CoreML/MLX adapters don't use the standard HF blob/snapshot cache layout
at all — they download real files (CoreML can't compile a symlinked
weight.bin) or store a converted copy under
`model_cache_dir()/<kind>/<org>--<name>/`. `_bespoke_local_dir()` covers
that case by name convention alone, since it's the one thing every such
adapter shares.
"""

import os
import shutil
from pathlib import Path

from splat.paths import hf_home_dir, model_cache_dir


def hub_cache_dir() -> Path:
    override = os.environ.get("HF_HUB_CACHE")
    return Path(override).expanduser() if override else hf_home_dir() / "hub"


class HuggingFaceModelSource:
    name = "huggingface"

    def _bespoke_local_dir(self, model_id: str) -> Path | None:
        suffix = model_id.replace("/", "--")
        for candidate in model_cache_dir().glob(f"*/{suffix}"):
            if candidate.is_dir() and any(candidate.iterdir()):
                return candidate
        return None

    def _bespoke_repo_ids(self) -> set[str]:
        return {
            candidate.name.replace("--", "/", 1)
            for candidate in model_cache_dir().glob("*/*")
            if "--" in candidate.name and candidate.is_dir() and any(candidate.iterdir())
        }

    @staticmethod
    def _repo_root(path: Path) -> Path:
        """snapshot_download resolves to .../models--org--name/snapshots/<rev>;
        the whole repo dir is what `remove`/`size_on_disk` care about."""
        if path.parent.name == "snapshots" and path.parent.parent.name.startswith("models--"):
            return path.parent.parent
        return path

    def pull(self, model_id: str, *, revision: str | None = None) -> Path:
        from huggingface_hub import snapshot_download

        path = snapshot_download(repo_id=model_id, revision=revision)
        return Path(path)

    def is_cached(self, model_id: str) -> bool:
        return self.local_path(model_id) is not None

    @staticmethod
    def _hub_snapshot(model_id: str) -> Path | None:
        snapshots = hub_cache_dir() / f"models--{model_id.replace('/', '--')}" / "snapshots"
        return next((p for p in sorted(snapshots.glob("*")) if p.is_dir()), None)

    def local_path(self, model_id: str) -> Path | None:
        return self._hub_snapshot(model_id) or self._bespoke_local_dir(model_id)

    def remove(self, model_id: str) -> None:
        path = self.local_path(model_id)
        if path is None:
            return
        shutil.rmtree(self._repo_root(path), ignore_errors=True)

    def size_on_disk(self, model_id: str) -> int:
        path = self.local_path(model_id)
        if path is None:
            return 0
        # Snapshot entries are symlinks into blobs/; following them would
        # double-count every weight file.
        return sum(
            entry.stat().st_size
            for entry in self._repo_root(path).rglob("*")
            if entry.is_file() and not entry.is_symlink()
        )

    def list_cached(self) -> list[str]:
        names = self._bespoke_repo_ids()
        for repo in hub_cache_dir().glob("models--*"):
            if (repo / "snapshots").is_dir() and any((repo / "snapshots").iterdir()):
                names.add(repo.name.removeprefix("models--").replace("--", "/", 1))
        return sorted(names)
