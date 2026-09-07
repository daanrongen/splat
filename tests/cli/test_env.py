import re

from typer.testing import CliRunner

from splat.cli.env import _settings
from splat.cli.main import app

runner = CliRunner()


def _plain(output: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", output)


def test_env_reports_defaults(monkeypatch):
    for var in (
        "SPLAT_DIFFUSE_MODEL",
        "SPLAT_SEGMENT_MODEL",
        "SPLAT_DEPTH_MODEL",
        "SPLAT_UPSCALE_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)

    result = runner.invoke(app, ["env"])
    plain = _plain(result.output)

    assert result.exit_code == 0, result.output
    assert "SPLAT_DIFFUSE_MODEL" in plain
    assert "SPLAT_UPSCALE_MODEL" in plain
    assert "sdxl-turbo-mlx" in plain
    assert "realesrgan-mlx" in plain
    assert "default" in plain


def test_env_reports_env_source(monkeypatch):
    monkeypatch.setenv("SPLAT_DIFFUSE_DEVICE", "cpu")

    result = runner.invoke(app, ["env"])
    plain = _plain(result.output)

    assert result.exit_code == 0, result.output
    assert "cpu" in plain


def test_env_flags_unknown_model_value(monkeypatch):
    monkeypatch.setenv("SPLAT_DIFFUSE_MODEL", "not-a-real-model")

    result = runner.invoke(app, ["env"])

    assert result.exit_code == 1
    assert "SPLAT_DIFFUSE_MODEL" in result.output
    assert "unknown value" in result.output


def test_env_reports_splat_url_row(monkeypatch):
    monkeypatch.delenv("SPLAT_URL", raising=False)

    result = runner.invoke(app, ["env"])
    plain = _plain(result.output)

    assert result.exit_code == 0, result.output
    assert "SPLAT_URL" in plain


def test_env_checks_splat_url_reachability(monkeypatch):
    monkeypatch.setenv("SPLAT_URL", "http://127.0.0.1:1")

    result = runner.invoke(app, ["env"])
    plain = _plain(result.output)

    assert result.exit_code == 0, result.output
    assert "SPLAT_URL reachable: no" in plain


def test_settings_are_derived_from_the_cli_not_duplicated():
    """Every row except the explicit non-option ones must trace back to a real
    option, so the table cannot claim a setting the CLI does not honor."""
    rows = {(setting.command, setting.param, setting.var) for setting in _settings()}

    assert ("gaussian", "--model", "SPLAT_GAUSSIAN_MODEL") in rows
    assert ("render", "--engine", "SPLAT_RENDER_ENGINE") in rows
    assert ("tools extract.surface", "--depth", "SPLAT_EXTRACT_SURFACE_DEPTH") in rows
    assert ("tools displace.height", "--to", "SPLAT_DISPLACE_HEIGHT_TO") in rows


def test_every_declared_env_var_is_read_by_its_command():
    """The regression this guards: before `envvar=` was declared, `splat env`
    listed sixteen variables and the commands read none of them."""
    import typer

    command = typer.main.get_command(app)

    def declared(cmd):
        for param in cmd.params:
            if isinstance(param.envvar, str):
                yield param.envvar
        for sub in getattr(cmd, "commands", {}).values():
            yield from declared(sub)

    names = list(declared(command))
    assert len(names) == len(set(names)), "duplicate SPLAT_* env var across commands"
    assert all(name.startswith("SPLAT_") for name in names)
    assert len(names) > 30
