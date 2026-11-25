"""Wraps the Hugging Face Hub as a ModelSource, the only adapter that performs a network
call to fetch model weights. Every repo is a directory of real files at
`model_cache_dir()/<org>--<name>/` holding only the files a model needs. Files already in
the legacy hub cache are hardlinked in instead of downloaded again."""

import os
import shutil
from collections.abc import Callable
from fnmatch import fnmatchcase
from pathlib import Path

from splat.paths import hf_home_dir, model_cache_dir


def hub_cache_dir() -> Path:
    override = os.environ.get("HF_HUB_CACHE")
    return Path(override).expanduser() if override else hf_home_dir() / "hub"


class _NullFile:
    encoding = "utf-8"

    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        pass


def _progress_bar(on_progress: Callable[[int, int], None]):
    """A `tqdm_class` that draws nothing and reports the bar `snapshot_download`
    reconstructs files on (the second byte bar it creates). Not an HF subclass,
    so HF_HUB_DISABLE_PROGRESS_BARS does not silence it."""
    from tqdm import tqdm

    byte_bars: list[tqdm] = []

    class ProgressBar(tqdm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **{**kwargs, "file": _NullFile(), "disable": False})
            if self.unit == "B":
                byte_bars.append(self)

        def update(self, n=1):
            super().update(n)
            if len(byte_bars) > 1 and self is byte_bars[1]:
                on_progress(int(self.n), int(self.total or 0))

    return ProgressBar


def _files(root: Path) -> list[str]:
    return [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(root).parts
    ]


def _matches(names: list[str], patterns: tuple[str, ...]) -> bool:
    if not patterns:
        return bool(names)
    return all(any(fnmatchcase(name, pattern) for name in names) for pattern in patterns)


def _link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


class HuggingFaceModelSource:
    name = "huggingface"

    @staticmethod
    def _dir(model_id: str) -> Path:
        return model_cache_dir() / model_id.replace("/", "--")

    @staticmethod
    def _hub_repo(model_id: str) -> Path:
        return hub_cache_dir() / f"models--{model_id.replace('/', '--')}"

    def _adopt(self, model_id: str, files: tuple[str, ...]) -> None:
        snapshot = next(iter(sorted((self._hub_repo(model_id) / "snapshots").glob("*"))), None)
        if snapshot is None:
            return
        root = self._dir(model_id)
        for name in _files(snapshot):
            wanted = not files or any(fnmatchcase(name, pattern) for pattern in files)
            if wanted and not (root / name).exists():
                _link((snapshot / name).resolve(), root / name)

    def pull(
        self,
        model_id: str,
        *,
        files: tuple[str, ...] = (),
        on_progress: Callable[[int, int], None] | None = None,
    ) -> Path:
        from huggingface_hub import snapshot_download

        self._adopt(model_id, files)
        if self.is_cached(model_id, files):
            return self._dir(model_id)
        extra = {} if on_progress is None else {"tqdm_class": _progress_bar(on_progress)}
        snapshot_download(
            repo_id=model_id,
            local_dir=self._dir(model_id),
            allow_patterns=list(files) or None,
            **extra,
        )
        return self._dir(model_id)

    def is_cached(self, model_id: str, files: tuple[str, ...] = ()) -> bool:
        return self.local_path(model_id, files) is not None

    def local_path(self, model_id: str, files: tuple[str, ...] = ()) -> Path | None:
        root = self._dir(model_id)
        return root if root.is_dir() and _matches(_files(root), files) else None

    def remove(self, model_id: str) -> None:
        shutil.rmtree(self._dir(model_id), ignore_errors=True)

    def size_on_disk(self, model_id: str) -> int:
        root = self._dir(model_id)
        return sum((root / name).stat().st_size for name in _files(root)) if root.is_dir() else 0

    def list_cached(self) -> list[str]:
        return sorted(
            path.name.replace("--", "/", 1)
            for path in model_cache_dir().iterdir()
            if path.is_dir() and any(path.iterdir())
        )

    def list_legacy(self) -> dict[str, int]:
        return {
            repo.name.removeprefix("models--").replace("--", "/", 1): sum(
                entry.stat().st_size
                for entry in repo.rglob("*")
                if entry.is_file() and not entry.is_symlink()
            )
            for repo in sorted(hub_cache_dir().glob("models--*"))
            if any((repo / "snapshots").glob("*"))
        }

    def remove_legacy(self, model_id: str) -> None:
        shutil.rmtree(self._hub_repo(model_id), ignore_errors=True)
