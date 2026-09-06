from dataclasses import dataclass

import typer
from rich.console import Console
from rich.table import Table

from splat.cli._console import error
from splat.env import resolve_verbose


@dataclass(frozen=True)
class _Setting:
    command: str
    param: str
    var: str
    default: str
    catalog: dict | None = None  # model-name catalog to validate against, if any


def _settings() -> list[_Setting]:
    from splat.registry.depth import DEPTH_CATALOG
    from splat.registry.diffuse import DIFFUSION_CATALOG
    from splat.registry.gaussian import GAUSSIAN_CATALOG
    from splat.registry.mesh import MESH_PREDICTION_CATALOG
    from splat.registry.segment import SEGMENTATION_CATALOG
    from splat.registry.upscale import UPSCALE_CATALOG

    return [
        _Setting("diffuse", "--model", "SPLAT_DIFFUSE_MODEL", "sdxl-turbo-mlx", DIFFUSION_CATALOG),
        _Setting("diffuse", "--device", "SPLAT_DIFFUSE_DEVICE", "auto"),
        _Setting("diffuse", "--steps", "SPLAT_DIFFUSE_STEPS", ""),
        _Setting("diffuse", "--seed", "SPLAT_DIFFUSE_SEED", ""),
        _Setting("diffuse", "--negative", "SPLAT_DIFFUSE_NEGATIVE", ""),
        _Setting("segment", "--model", "SPLAT_SEGMENT_MODEL", "sam-mlx", SEGMENTATION_CATALOG),
        _Setting("segment", "--device", "SPLAT_SEGMENT_DEVICE", "auto"),
        _Setting("segment", "--max-stickers", "SPLAT_SEGMENT_MAX_STICKERS", "20"),
        _Setting("depth", "--model", "SPLAT_DEPTH_MODEL", "depth-pro", DEPTH_CATALOG),
        _Setting("depth", "--device", "SPLAT_DEPTH_DEVICE", "auto"),
        _Setting("upscale", "--model", "SPLAT_UPSCALE_MODEL", "realesrgan-mlx", UPSCALE_CATALOG),
        _Setting("upscale", "--factor", "SPLAT_UPSCALE_FACTOR", "4"),
        _Setting("upscale", "--tile", "SPLAT_UPSCALE_TILE", "0"),
        _Setting("gaussian", "--model", "SPLAT_GAUSSIAN_MODEL", "mvsplat", GAUSSIAN_CATALOG),
        _Setting("gaussian", "--device", "SPLAT_GAUSSIAN_DEVICE", "auto"),
        _Setting("mesh", "--model", "SPLAT_MESH_MODEL", "triposr", MESH_PREDICTION_CATALOG),
        _Setting("mesh", "--device", "SPLAT_MESH_DEVICE", "auto"),
        _Setting("http", "--host", "SPLAT_HOST", "127.0.0.1:8000"),
        _Setting("tools compress", "--profile", "SPLAT_COMPRESS_PROFILE", "web-delivery"),
        _Setting("validate", "--strict", "SPLAT_VALIDATE_STRICT", "false"),
        _Setting("tools displace.height", "--to", "SPLAT_DISPLACE_HEIGHT_TO", "glb"),
        _Setting("*", "(client)", "SPLAT_URL", ""),
    ]


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
