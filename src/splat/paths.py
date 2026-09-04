"""Cache directory resolution. mise.toml sets SPLAT_MODEL_CACHE_DIR/
SPLAT_ASSET_CACHE_DIR explicitly for local dev; these fallbacks cover
non-mise contexts (a Homebrew install, CI) using the XDG Base Directory
convention. HF_HOME defaults to huggingface_hub's own standard location
(`$XDG_CACHE_HOME/huggingface`), not a splat-specific one — raw model
downloads are shared with every other tool on the machine, not siloed.
"""

import os
from pathlib import Path


def _xdg_cache_home() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")


def _resolve(env_var: str, default: Path) -> Path:
    path = Path(os.environ[env_var]) if os.environ.get(env_var) else default
    path.mkdir(parents=True, exist_ok=True)
    return path


def hf_home_dir() -> Path:
    return _resolve("HF_HOME", _xdg_cache_home() / "huggingface")


def model_cache_dir() -> Path:
    return _resolve("SPLAT_MODEL_CACHE_DIR", _xdg_cache_home() / "splat" / "models")


def asset_cache_dir() -> Path:
    return _resolve("SPLAT_ASSET_CACHE_DIR", _xdg_cache_home() / "splat" / "assets")
