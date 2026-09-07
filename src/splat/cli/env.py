"""`splat env` reads the Typer app itself rather than keeping a second copy of
every default. Anything with `envvar=` on its option shows up here
automatically, so the table cannot drift from the command signatures.
"""

from dataclasses import dataclass

import click
import typer
from rich.console import Console
from rich.table import Table

from splat.cli._console import error
from splat.env import resolve_verbose

# Env vars that aren't command options: cache roots, the client redirect, and
# the two the Blender adapter reads directly.
_NON_OPTION_VARS: list[tuple[str, str, str, str]] = [
    ("*", "(client)", "SPLAT_URL", ""),
    ("*", "(cache)", "SPLAT_MODEL_CACHE_DIR", ""),
    ("*", "(cache)", "SPLAT_ASSET_CACHE_DIR", ""),
    ("render", "(adapter)", "SPLAT_BLENDER_BIN", ""),
    ("render", "(adapter)", "SPLAT_RENDER_TIMEOUT", "1800"),
]


@dataclass(frozen=True)
class _Setting:
    command: str
    param: str
    var: str
    default: str
    catalog: dict | None = None  # model-name catalog to validate against, if any


def _catalog_for(command: str, param: str) -> dict | None:
    if param != "--model":
        return None
    from splat.registry.caption import CAPTION_CATALOG
    from splat.registry.depth import DEPTH_CATALOG
    from splat.registry.diffuse import DIFFUSION_CATALOG
    from splat.registry.embed import EMBEDDING_CATALOG
    from splat.registry.gaussian import GAUSSIAN_CATALOG
    from splat.registry.segment import SEGMENTATION_CATALOG
    from splat.registry.upscale import UPSCALE_CATALOG

    return {
        "diffuse": DIFFUSION_CATALOG,
        "caption": CAPTION_CATALOG,
        "embed": EMBEDDING_CATALOG,
        "segment": SEGMENTATION_CATALOG,
        "depth": DEPTH_CATALOG,
        "upscale": UPSCALE_CATALOG,
        "gaussian": GAUSSIAN_CATALOG,
    }.get(command)


def _long_opt(param: click.Parameter) -> str:
    """Prefer `--to` over `-t`; short flags read as noise in this table."""
    return next((opt for opt in param.opts if opt.startswith("--")), param.opts[0])


def _default_text(param: click.Parameter) -> str:
    if isinstance(param.default, bool):
        return str(param.default).lower()
    return "" if param.default is None else str(param.default)


def _walk(command: click.Command, path: tuple[str, ...]) -> list[_Setting]:
    name = " ".join(path)
    settings = [
        _Setting(
            command=name,
            param=_long_opt(param),
            var=param.envvar,
            default=_default_text(param),
            catalog=_catalog_for(name, _long_opt(param)),
        )
        for param in command.params
        if isinstance(param.envvar, str)
    ]
    for sub_name, sub in getattr(command, "commands", {}).items():
        settings.extend(_walk(sub, (*path, sub_name)))
    return settings


def _settings() -> list[_Setting]:
    from splat.cli.main import app

    settings = _walk(typer.main.get_command(app), ())
    settings.extend(
        _Setting(command=command, param=param, var=var, default=default)
        for command, param, var, default in _NON_OPTION_VARS
    )
    return settings


def _reachable(host: str) -> bool:
    import httpx

    try:
        httpx.get(f"{host.rstrip('/')}/openapi.json", timeout=2.0)
        return True
    except httpx.HTTPError:
        return False


def env() -> None:
    """Print every SPLAT_* setting's resolved value and source (env|mise|default)."""
    table = Table(title="splat environment-variable defaults")
    table.add_column("command")
    table.add_column("param")
    table.add_column("env var")
    table.add_column("value")
    table.add_column("source")

    invalid: list[_Setting] = []
    for setting in _settings():
        resolved = resolve_verbose(setting.var, setting.default)
        value = resolved.value or "[dim](none)[/dim]"
        if setting.catalog is not None and resolved.value and resolved.value not in setting.catalog:
            value = f"[red]{resolved.value}[/red]"
            invalid.append(setting)
        table.add_row(setting.command, setting.param, setting.var, value, resolved.source)

    Console(width=120).print(table)

    url = resolve_verbose("SPLAT_URL", "").value
    if url:
        Console(width=120).print(f"SPLAT_URL reachable: {'yes' if _reachable(url) else 'no'}")

    if invalid:
        for setting in invalid:
            available = ", ".join(sorted(setting.catalog))
            error(f"{setting.var}: unknown value, available: {available}")
        raise typer.Exit(code=1)
