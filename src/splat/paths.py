"""Cache directory resolution. mise.toml sets HF_HOME/SPLAT_MODEL_CACHE_DIR/
SPLAT_ASSET_CACHE_DIR explicitly for local dev; these fallbacks cover
non-mise contexts (a Homebrew install, CI) using the XDG Base Directory
convention — everything under one `$XDG_CACHE_HOME/splat/` root rather than
scattered per-adapter defaults.
"""

import os
from pathlib import Path


def _xdg_cache_home() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")


def _resolve(env_var: str, *default_parts: str) -> Path:
    if os.environ.get(env_var):
        path = Path(os.environ[env_var])
    else:
        path = _xdg_cache_home() / "splat" / Path(*default_parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def hf_home_dir() -> Path:
    return _resolve("HF_HOME", "huggingface")


def model_cache_dir() -> Path:
    return _resolve("SPLAT_MODEL_CACHE_DIR", "models")


def asset_cache_dir() -> Path:
    return _resolve("SPLAT_ASSET_CACHE_DIR", "assets")
