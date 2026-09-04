from typer.testing import CliRunner

from splat.cli.main import app

runner = CliRunner()


def test_http_defaults_to_loopback_8000(mocker):
    run = mocker.patch("uvicorn.run")

    result = runner.invoke(app, ["http"])

    assert result.exit_code == 0, result.output
    run.assert_called_once()
    assert run.call_args.kwargs["host"] == "127.0.0.1"
    assert run.call_args.kwargs["port"] == 8000


def test_http_parses_host_and_port_from_one_flag(mocker):
    run = mocker.patch("uvicorn.run")

    result = runner.invoke(app, ["http", "--host", "0.0.0.0:9000"])

    assert result.exit_code == 0, result.output
    assert run.call_args.kwargs["host"] == "0.0.0.0"
    assert run.call_args.kwargs["port"] == 9000
