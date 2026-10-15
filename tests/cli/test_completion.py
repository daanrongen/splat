from typer.testing import CliRunner

from splat.cli.main import app

runner = CliRunner()


def test_prints_the_zsh_script():
    result = runner.invoke(app, ["completion", "zsh"])

    assert result.exit_code == 0
    assert result.output.splitlines()[0] == "#compdef splat"


def test_no_longer_offers_install_completion():
    assert "--install-completion" not in runner.invoke(app, ["--help"]).output
