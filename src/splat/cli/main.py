import os

from splat.paths import asset_cache_dir, hf_home_dir, model_cache_dir

# Must run before any transitive `huggingface_hub` import below (it reads
# HF_HOME at import time) — guarantees the XDG-consolidated cache applies
# even when mise's own env activation isn't in effect for this process.
os.environ.setdefault("HF_HOME", str(hf_home_dir()))
os.environ.setdefault("SPLAT_MODEL_CACHE_DIR", str(model_cache_dir()))
os.environ.setdefault("SPLAT_ASSET_CACHE_DIR", str(asset_cache_dir()))

import typer

from splat.cli.caption import caption
from splat.cli.depth import depth
from splat.cli.diffuse import diffuse
from splat.cli.env import env
from splat.cli.gaussian import gaussian
from splat.cli.http import http
from splat.cli.info import info
from splat.cli.mcp import mcp
from splat.cli.mesh import mesh
from splat.cli.models import models_app
from splat.cli.segment import segment
from splat.cli.tools import tools_app
from splat.cli.train import train
from splat.cli.upscale import upscale
from splat.cli.validate import validate

app = typer.Typer(
    help=(
        "splat runs stochastic 3D image-space backends at top level; deterministic "
        "transforms live under `splat tools`."
    ),
    no_args_is_help=True,
)

app.command("diffuse")(diffuse)
app.command("caption")(caption)
app.command("segment")(segment)
app.command("depth")(depth)
app.command("upscale")(upscale)
app.command("gaussian")(gaussian)
app.command("mesh")(mesh)
app.command("train")(train)
app.add_typer(tools_app, name="tools")
app.command("info")(info)
app.command("validate")(validate)
app.add_typer(models_app, name="models")
app.command("http")(http)
app.command("mcp")(mcp)
app.command("env")(env)


if __name__ == "__main__":
    app()
