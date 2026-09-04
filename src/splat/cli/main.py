import os

from splat.paths import asset_cache_dir, hf_home_dir, model_cache_dir

# Must run before any transitive `huggingface_hub` import below (it reads
# HF_HOME at import time) — guarantees the XDG-consolidated cache applies
# even when mise's own env activation isn't in effect for this process.
os.environ.setdefault("HF_HOME", str(hf_home_dir()))
os.environ.setdefault("SPLAT_MODEL_CACHE_DIR", str(model_cache_dir()))
os.environ.setdefault("SPLAT_ASSET_CACHE_DIR", str(asset_cache_dir()))

import typer

from splat.cli.compress import compress
from splat.cli.convert import convert
from splat.cli.depth import depth
from splat.cli.diffuse import diffuse
from splat.cli.env import env
from splat.cli.gaussian import gaussian
from splat.cli.http import http
from splat.cli.info import info
from splat.cli.mesh import mesh
from splat.cli.models import models_app
from splat.cli.segment import segment
from splat.cli.tools import tools_app
from splat.cli.train import train
from splat.cli.validate import validate

app = typer.Typer(
    help="splat — a universal converter for Gaussian Splatting files and models.",
    no_args_is_help=True,
)

app.command("convert")(convert)
app.command("info")(info)
app.command("validate")(validate)
app.command("compress")(compress)
app.command("mesh")(mesh)
app.command("train")(train)
app.command("diffuse")(diffuse)
app.command("segment")(segment)
app.command("depth")(depth)
app.command("gaussian")(gaussian)
app.command("http")(http)
app.command("env")(env)
app.add_typer(models_app, name="models")
app.add_typer(tools_app, name="tools")


if __name__ == "__main__":
    app()
