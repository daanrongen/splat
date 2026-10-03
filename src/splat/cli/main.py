import os

from splat.paths import hf_home_dir, manifest_cache_dir, model_cache_dir

# Must run before any transitive `huggingface_hub` import below (it reads
# HF_HOME at import time) — guarantees the XDG-consolidated cache applies
# even when mise's own env activation isn't in effect for this process.
os.environ.setdefault("HF_HOME", str(hf_home_dir()))
os.environ.setdefault("SPLAT_MODEL_CACHE_DIR", str(model_cache_dir()))
os.environ.setdefault("SPLAT_MANIFEST_CACHE_DIR", str(manifest_cache_dir()))

import typer

from splat.cli._lazy import LazyCommandSpec, LazyTyperGroup

LazyTyperGroup.lazy_commands = {
    "diffuse": LazyCommandSpec(
        "splat.cli.diffuse", "diffuse", "command", "Generate or edit an image asset."
    ),
    "caption": LazyCommandSpec(
        "splat.cli.caption", "caption", "command", "Caption image or sticker assets."
    ),
    "embed": LazyCommandSpec(
        "splat.cli.embed", "embed", "command", "Embed image, sticker, caption, or text assets."
    ),
    "segment": LazyCommandSpec(
        "splat.cli.segment", "segment", "command", "Segment images into RGBA stickers."
    ),
    "depth": LazyCommandSpec(
        "splat.cli.depth", "depth", "command", "Estimate per-pixel depth maps."
    ),
    "upscale": LazyCommandSpec("splat.cli.upscale", "upscale", "command", "Upscale raster assets."),
    "gaussian": LazyCommandSpec(
        "splat.cli.gaussian", "gaussian", "command", "Reconstruct Gaussian splats."
    ),
    "render": LazyCommandSpec(
        "splat.cli.render", "render", "command", "Render Gaussian splats to images."
    ),
    "train": LazyCommandSpec(
        "splat.cli.train", "train", "command", "Train or refine per-scene Gaussian splats."
    ),
    "tools": LazyCommandSpec("splat.cli.tools", "tools_app", "typer", "Deterministic transforms."),
    "info": LazyCommandSpec("splat.cli.info", "info", "command", "Inspect a Gaussian splat file."),
    "validate": LazyCommandSpec(
        "splat.cli.validate", "validate", "command", "Validate a Gaussian splat file."
    ),
    "models": LazyCommandSpec(
        "splat.cli.models", "models_app", "typer", "Manage locally cached model weights."
    ),
    "manifest": LazyCommandSpec(
        "splat.cli.manifest", "manifest_app", "typer", "Inspect and manage cached manifests."
    ),
    "http": LazyCommandSpec("splat.cli.http", "http", "command", "Run the local HTTP server."),
    "mcp": LazyCommandSpec("splat.cli.mcp", "mcp", "command", "Run the MCP stdio server."),
    "env": LazyCommandSpec("splat.cli.env", "env", "command", "Show SPLAT_* environment settings."),
}

app = typer.Typer(
    help=(
        "splat runs backends with a swappable model/engine catalog (--model) at top "
        "level; single fixed-algorithm operations with no catalog live under `splat tools`."
    ),
    no_args_is_help=True,
    cls=LazyTyperGroup,
)


def _print_version(value: bool) -> None:
    if value:
        from importlib.metadata import version

        typer.echo(version("splat"))
        raise typer.Exit()


@app.callback()
def _main(
    _version: bool = typer.Option(
        False, "--version", callback=_print_version, is_eager=True, help="Show the version."
    ),
) -> None:
    """splat runs model-backed stages, deterministic tools, and local cache admin."""


if __name__ == "__main__":
    app()
