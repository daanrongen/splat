import typer
from typer._completion_shared import Shells, get_completion_script


def completion(
    shell: Shells = typer.Argument(..., help="Shell to generate the script for."),
) -> None:
    """Print a shell completion script; redirect it to a file, nothing is installed."""
    typer.echo(
        get_completion_script(prog_name="splat", complete_var="_SPLAT_COMPLETE", shell=shell.value)
    )
