from typer.testing import CliRunner

from splat.cli.main import app
from splat.domain.value_objects import MIT
from tests.handlers.test_diffuse import FakeDiffusionBackend
from tests.image_helpers import write_sample_png

runner = CliRunner()


def _patch_diffuse(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    backend = FakeDiffusionBackend(MIT)
    mocker.patch("splat.handlers.diffuse.get_diffusion_backend", return_value=backend)
    return backend


def test_diffuse_creates_missing_output_directories(mocker, tmp_path, monkeypatch):
    _patch_diffuse(mocker, tmp_path, monkeypatch)
    output = tmp_path / "missing" / "dir" / "x.png"

    result = runner.invoke(app, ["diffuse", "a fox", "--model", "fake-diffuser", "-o", str(output)])

    assert result.exit_code == 0, result.output
    assert output.exists()


def test_diffuse_rejects_unwritable_output_before_generating(mocker, tmp_path, monkeypatch):
    backend = _patch_diffuse(mocker, tmp_path, monkeypatch)
    blocker = write_sample_png(tmp_path / "blocker.png")

    result = runner.invoke(
        app, ["diffuse", "a fox", "--model", "fake-diffuser", "-o", str(blocker / "x.png")]
    )

    assert result.exit_code == 1
    assert backend.last_call is None
    assert "Traceback" not in result.output
