import pytest
from typer.testing import CliRunner

from splat.cli import doctor as doctor_cli
from splat.cli.main import app
from splat.domain.errors import RenderBackendError

runner = CliRunner()


def test_a_broken_runtime_fails_without_stopping_the_other_checks(mocker, monkeypatch):
    monkeypatch.delenv("SPLAT_URL", raising=False)
    mocker.patch.object(doctor_cli, "_RUNTIMES", ("torch", "no_such_runtime"))
    mocker.patch.object(doctor_cli, "_device", return_value="cpu")
    mocker.patch.object(doctor_cli, "_models", return_value="0 of 1 cached")

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "fail  no_such_runtime  ModuleNotFoundError" in result.output
    assert "ok    device  cpu" in result.output
    assert "ok    models  0 of 1 cached" in result.output


def test_passes_when_every_check_does(mocker, monkeypatch):
    monkeypatch.delenv("SPLAT_URL", raising=False)
    mocker.patch.object(doctor_cli, "_RUNTIMES", ("json",))
    mocker.patch.object(doctor_cli, "_device", return_value="cpu")
    mocker.patch.object(doctor_cli, "_models", return_value="1 of 1 cached")

    assert runner.invoke(app, ["doctor"]).exit_code == 0


def test_checks_the_server_when_splat_url_is_set(mocker, monkeypatch):
    monkeypatch.setenv("SPLAT_URL", "http://splat.test")
    mocker.patch.object(doctor_cli, "_RUNTIMES", ())
    mocker.patch.object(doctor_cli, "_device", return_value="cpu")
    mocker.patch.object(doctor_cli, "_models", return_value="1 of 1 cached")
    mocker.patch.object(doctor_cli, "_server", side_effect=ConnectionError("refused"))

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "fail  server  ConnectionError: refused" in result.output


def _server_answers(mocker, path):
    response = mocker.Mock()
    response.raise_for_status.return_value.json.return_value = {"path": path}
    mocker.patch("httpx.get", return_value=response)


def test_server_blender_reports_the_servers_path(mocker, monkeypatch):
    monkeypatch.setenv("SPLAT_URL", "http://splat.test")
    _server_answers(mocker, "/opt/homebrew/bin/blender")

    assert doctor_cli._server_blender() == "/opt/homebrew/bin/blender"


def test_server_blender_fails_when_the_server_has_none(mocker, monkeypatch):
    monkeypatch.setenv("SPLAT_URL", "http://splat.test")
    _server_answers(mocker, None)

    with pytest.raises(RenderBackendError, match="server's PATH"):
        doctor_cli._server_blender()


def test_checks_the_servers_blender_when_splat_url_is_set(monkeypatch):
    monkeypatch.setenv("SPLAT_URL", "http://splat.test")

    assert "server blender" in [name for name, _ in doctor_cli._checks()]
