import re

from typer.testing import CliRunner

from splat.cli.main import app

runner = CliRunner()


def _plain(output: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", output)


def test_env_reports_defaults(monkeypatch):
    for var in ("SPLAT_DIFFUSE_MODEL", "SPLAT_SEGMENT_MODEL", "SPLAT_DEPTH_MODEL"):
        monkeypatch.delenv(var, raising=False)

    result = runner.invoke(app, ["env"])
    plain = _plain(result.output)

    assert result.exit_code == 0, result.output
    assert "SPLAT_DIFFUSE_MODEL" in plain
    assert "sdxl-turbo-mlx" in plain
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
