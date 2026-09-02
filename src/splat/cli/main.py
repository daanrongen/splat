import typer

from splat.cli.compress import compress
from splat.cli.convert import convert
from splat.cli.info import info
from splat.cli.mesh import export
from splat.cli.models import models_app
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
app.command("export")(export)
app.command("train")(train)
app.add_typer(models_app, name="models")


if __name__ == "__main__":
    app()
