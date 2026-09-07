"""SPLAT_<COMMAND>_<PARAM> default resolution, sibling to paths.py's cache-dir
resolution. Precedence: CLI flag (handled by Click via `envvar=`, not here) >
plain `os.environ` (covers shell exports and mise-hooked shells alike) > `mise
env --json` for invocations without the shell hook active > built-in default.

Click only ever reads `os.environ`, so `seed_from_mise()` materializes the mise
layer there once at startup rather than leaving a second resolution path for
`splat env` to report on and nothing else to honor.
"""

import json
import os
import shutil
import subprocess
from dataclasses import dataclass

_mise_env_cache: dict[str, str] | None = None


def _mise_env() -> dict[str, str]:
    global _mise_env_cache
    if _mise_env_cache is not None:
        return _mise_env_cache

    _mise_env_cache = {}
    if shutil.which("mise") is None:
        return _mise_env_cache

    try:
        result = subprocess.run(
            ["mise", "env", "--json"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        _mise_env_cache = json.loads(result.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        pass
    return _mise_env_cache


_mise_seeded: set[str] = set()


def seed_from_mise() -> None:
    """Copy mise's SPLAT_* vars into os.environ so Click's `envvar=` defaults
    see them. Real environment variables always win."""
    for name, value in _mise_env().items():
        if name.startswith("SPLAT_") and name not in os.environ:
            os.environ[name] = value
            _mise_seeded.add(name)


@dataclass(frozen=True)
class Resolved:
    value: str
    source: str  # "env" | "mise" | "default"


def resolve_verbose(var_name: str, default: str) -> Resolved:
    if var_name in os.environ:
        return Resolved(os.environ[var_name], "mise" if var_name in _mise_seeded else "env")
    mise_value = _mise_env().get(var_name)
    if mise_value is not None:
        return Resolved(mise_value, "mise")
    return Resolved(default, "default")


def resolve(var_name: str, default: str) -> str:
    return resolve_verbose(var_name, default).value
