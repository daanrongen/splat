"""`.env.example` is generated, not written. These guard the generation and the
committed file against each other, since a stale example file was the original
problem: it listed 2 of 48 settings, a `SPLAT_MODEL_CACHE_DIR` value that
matched neither `paths.py` nor `mise.toml`, and `HF_TOKEN` with no explanation.
"""

from pathlib import Path

from typer.testing import CliRunner

from splat.cli.env import _export, _settings
from splat.cli.main import app

runner = CliRunner()

ENV_EXAMPLE = Path(__file__).resolve().parents[2] / ".env.example"


def test_env_example_matches_the_cli():
    assert ENV_EXAMPLE.read_text() == _export(), (
        "`.env.example` is stale; regenerate it with `splat env --export > .env.example`"
    )


def test_export_lists_every_setting_the_table_shows():
    exported = _export()
    for setting in _settings():
        assert f"# {setting.var}=" in exported, setting.var


def test_export_comments_out_every_setting():
    """An uncommented line would silently take effect the moment the file is
    copied to .env, which is the opposite of a template."""
    for line in _export().splitlines():
        assert not line or line.startswith("#"), line


def test_export_cli_flag_writes_the_template():
    result = runner.invoke(app, ["env", "--export"])

    assert result.exit_code == 0, result.output
    assert "SPLAT_DIFFUSE_MODEL=sdxl-turbo-mlx" in result.output
    assert "# --- gaussian ---" in result.output
