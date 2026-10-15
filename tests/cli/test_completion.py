import os
import subprocess
import sys

from typer.testing import CliRunner

from splat.cli.main import app

runner = CliRunner()


def test_prints_the_zsh_script():
    result = runner.invoke(app, ["completion", "zsh"])

    assert result.exit_code == 0
    assert result.output.splitlines()[0] == "#compdef splat"


def test_no_longer_offers_install_completion():
    assert "--install-completion" not in runner.invoke(app, ["--help"]).output


def test_completes_command_names():
    code = 'import sys; from splat.cli.main import main; sys.argv[0] = "splat"; main()'
    env = {**os.environ, "_SPLAT_COMPLETE": "complete_zsh", "_TYPER_COMPLETE_ARGS": "splat diffu"}

    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)

    assert "diffuse" in result.stdout
