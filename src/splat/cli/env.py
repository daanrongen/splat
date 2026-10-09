"""`splat env` reads the Typer app itself rather than keeping a second copy of
every default. Anything with `envvar=` on its option shows up here
automatically, so neither the table nor `--export` can drift from the command
signatures.

Nothing here participates in resolution: Click reads `envvar=` off os.environ,
and getting values into os.environ is mise's job (its `[env]` block and
`_.file = [".env"]`) or uv's (`uv run --env-file`). This module only reports.
"""

import os
from dataclasses import dataclass

import click
import typer

from splat.cli import _table as table
from splat.cli._console import console, error

# Env vars that aren't command options: cache roots, the client redirect, and
# the two the Blender adapter reads directly. The comment is what `--export`
# writes above each one, since a bare name explains nothing in a .env file.
_NON_OPTION_VARS: list[tuple[str, str, str, str, str]] = [
    (
        "global",
        "client",
        "SPLAT_URL",
        "",
        "Remote `splat http` base URL; unset runs everything locally",
    ),
    (
        "global",
        "cache",
        "SPLAT_MODEL_CACHE_DIR",
        "",
        "Converted/compiled model artifacts; defaults to $XDG_CACHE_HOME/splat/models",
    ),
    (
        "global",
        "cache",
        "SPLAT_MANIFEST_CACHE_DIR",
        "",
        "One file per manifest; defaults to $XDG_CACHE_HOME/splat/manifests",
    ),
    ("render", "adapter", "SPLAT_BLENDER_BIN", "", "Blender executable when it isn't on PATH"),
    (
        "render",
        "adapter",
        "SPLAT_RENDER_TIMEOUT",
        "1800",
        "Seconds before a render is killed; 0 disables",
    ),
]


@dataclass(frozen=True)
class _Setting:
    command: str
    param: str
    var: str
    default: str
    help: str = ""
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
            help=param.help or "",
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
        _Setting(command=command, param=f"({param})", var=var, default=default, help=help_text)
        for command, param, var, default, help_text in _NON_OPTION_VARS
    )
    # Regroup so each command's rows are contiguous: `render` has both options
    # and adapter-level vars, and the latter are appended after the whole walk.
    order = list(dict.fromkeys(setting.command for setting in settings))
    return sorted(settings, key=lambda setting: order.index(setting.command))


def _resolve(var: str, default: str) -> tuple[str, str]:
    """(value, origin). "env" covers a shell export, mise's `[env]`, and
    `.env` alike - by the time Click looks, they are indistinguishable, and
    pretending otherwise meant shelling out to `mise env --json` per run."""
    if var in os.environ:
        return os.environ[var], "env"
    return default, "default"


def _reachable(host: str) -> str | None:
    """ "The server version, "" if it reports none, None if unreachable."""
    import httpx

    try:
        response = httpx.get(f"{host.rstrip('/')}/version", timeout=2.0)
    except httpx.HTTPError:
        return None
    return response.json()["version"] if response.status_code == 200 else ""


def _export() -> str:
    """A dotenv template, grouped by command, every line commented out with its
    built-in default as the value. Generated from the same walk as the table so
    `.env.example` cannot drift from the CLI either."""
    lines = [
        "# Every SPLAT_* setting `splat` reads, generated by `splat env --export`.",
        "#",
        "# Regenerate with `splat env --export > .env.example`; `mise run check` fails",
        "# if this file drifts from the CLI. Copy it to .env (gitignored) and uncomment",
        "# what you want to change - every value shown is already the built-in default.",
        "#",
        "# mise loads .env via `_.file` in mise.toml, so an uncommented line applies to",
        "# every `splat` run in an activated shell. Outside mise, either export the vars",
        "# yourself or use `uv run --env-file .env splat ...`.",
        "#",
        "# Precedence: CLI flag > environment > built-in default. `splat env` shows what",
        "# each setting currently resolves to and where it came from.",
    ]
    current = None
    for setting in _settings():
        if setting.command != current:
            current = setting.command
            lines.extend(["", f"# --- {current} ---"])
        if setting.help:
            lines.append(f"# {setting.help}")
        lines.append(f"# {setting.var}={setting.default}")

    lines.extend(
        [
            "",
            "# --- not read by splat itself ---",
            "# Passed through to huggingface_hub for gated or private model repos.",
            "# HF_TOKEN=",
        ]
    )
    return "\n".join(lines) + "\n"


def env(
    export: bool = typer.Option(
        False, "--export", help="Print a .env template instead of the resolved table."
    ),
) -> None:
    """Print every SPLAT_* setting's resolved value and where it came from."""
    if export:
        print(_export(), end="")
        return

    invalid: list[_Setting] = []
    rows = []
    for setting in _settings():
        resolved, origin = _resolve(setting.var, setting.default)
        value = resolved or "[dim]-[/dim]"
        if setting.catalog is not None and resolved and resolved not in setting.catalog:
            value = f"[red]{resolved}[/red]"
            invalid.append(setting)
        rows.append((setting.command, setting.param, setting.var, value, origin))

    # The command is a row grouping rather than a column: it is already the
    # middle of every var name, and `splat <cmd> --help` names the var next to
    # its own option, so that width is better spent on the value.
    table.render(
        ("command", "param", "env var", "value", "from"),
        rows,
        flex=3,
        group_by=0,
        caption="grouped by command  ·  `splat env --export` writes a .env template",
    )

    url = os.environ.get("SPLAT_URL", "")
    if url:
        server = _reachable(url)
        detail = "no" if server is None else f"yes (server {server})" if server else "yes"
        console.print(f"SPLAT_URL reachable: {detail}")

    if invalid:
        for setting in invalid:
            available = ", ".join(sorted(setting.catalog))
            error(f"{setting.var}: unknown value, available: {available}")
        raise typer.Exit(code=1)
