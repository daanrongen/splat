import importlib
import os
import platform
from collections.abc import Callable

import typer

from splat import __version__
from splat.cli._console import console

_RUNTIMES = ("torch", "torchvision", "mlx.core", "coremltools", "transformers")


def _import(module: str) -> str:
    return getattr(importlib.import_module(module), "__version__", "ok")


def _device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def _models() -> str:
    from splat.registry.wiring import get_client

    rows = get_client().models_list()
    return f"{sum(row.cached for row in rows)} of {len(rows)} cached"


def _server() -> str:
    import httpx

    url = os.environ["SPLAT_URL"].rstrip("/")
    token = os.environ.get("SPLAT_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else None
    version = httpx.get(f"{url}/version", timeout=2.0).raise_for_status().json()["version"]
    httpx.get(f"{url}/models", headers=headers, timeout=5.0).raise_for_status()
    return f"{url} runs splat {version}"


def _checks() -> list[tuple[str, Callable[[], str]]]:
    checks: list[tuple[str, Callable[[], str]]] = [
        ("splat", lambda: f"{__version__}, python {platform.python_version()}"),
        *((module, lambda module=module: _import(module)) for module in _RUNTIMES),
        ("device", _device),
        ("models", _models),
    ]
    if os.environ.get("SPLAT_URL"):
        checks.append(("server", _server))
    return checks


def doctor() -> None:
    """Check the install: runtimes import, the device, cached models, and SPLAT_URL."""
    failed = 0
    for name, check in _checks():
        try:
            console.print(f"[green]ok[/green]    {name}  {check()}", highlight=False)
        except Exception as exc:
            failed += 1
            reason = f"{type(exc).__name__}: {exc}".splitlines()[0]
            console.print(f"[red]fail[/red]  {name}  {reason}", highlight=False)
    if failed:
        raise typer.Exit(code=1)
