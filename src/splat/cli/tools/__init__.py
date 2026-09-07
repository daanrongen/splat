import typer

from splat.cli.tools.compress import compress
from splat.cli.tools.convert import convert
from splat.cli.tools.declutter import declutter
from splat.cli.tools.displace_height import displace_height
from splat.cli.tools.extract_surface import extract_surface
from splat.cli.tools.normalize_color import normalize_color

tools_app = typer.Typer(
    help=(
        "Deterministic transforms: convert, compress, declutter, normalize.color, displace, "
        "and extract."
    ),
    no_args_is_help=True,
)

tools_app.command("convert")(convert)
tools_app.command("compress")(compress)
tools_app.command("declutter")(declutter)
tools_app.command("normalize.color")(normalize_color)
tools_app.command("displace.height")(displace_height)
tools_app.command("extract.surface")(extract_surface)
