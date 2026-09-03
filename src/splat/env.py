"""SPLAT_<COMMAND>_<PARAM> default resolution, sibling to paths.py's cache-dir
resolution. Precedence: CLI flag (handled by the caller, not here) > plain
`os.environ` (covers shell exports and mise-hooked shells alike) > `mise env
--json` for invocations without the shell hook active > built-in default.
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


@dataclass(frozen=True)
class Resolved:
    value: str
    source: str  # "env" | "mise" | "default"


def resolve_verbose(var_name: str, default: str) -> Resolved:
    if var_name in os.environ:
        return Resolved(os.environ[var_name], "env")
    mise_value = _mise_env().get(var_name)
    if mise_value is not None:
        return Resolved(mise_value, "mise")
    return Resolved(default, "default")


def resolve(var_name: str, default: str) -> str:
    return resolve_verbose(var_name, default).value
