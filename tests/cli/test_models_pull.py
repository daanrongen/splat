from io import StringIO

import pytest
from rich.console import Console
from typer.testing import CliRunner

from splat.cli.main import app
from splat.domain.errors import SplatDomainError

runner = CliRunner()


class FakeClient:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    def models_pull(self, name, *, on_progress=None):
        if self.error:
            raise self.error
        if on_progress is not None:
            on_progress("org/model", 50, 100)
            on_progress("org/model", 100, 100)


@pytest.fixture
def stderr(mocker):
    def use(*, terminal: bool) -> StringIO:
        buffer = StringIO()
        console = Console(file=buffer, force_terminal=terminal, width=100)
        mocker.patch("splat.cli.models.err_console", console)
        return buffer

    return use


def test_pull_hands_the_client_a_progress_callback_on_a_terminal(mocker, stderr):
    stderr(terminal=True)
    client = FakeClient()
    spy = mocker.spy(client, "models_pull")
    mocker.patch("splat.cli.models.get_client", return_value=client)

    runner.invoke(app, ["models", "pull", "depth-pro"])

    assert spy.call_args.kwargs["on_progress"] is not None


def test_pull_draws_no_bar_when_stderr_is_not_a_terminal(mocker, stderr):
    buffer = stderr(terminal=False)
    mocker.patch("splat.cli.models.get_client", return_value=FakeClient())

    result = runner.invoke(app, ["models", "pull", "depth-pro"])

    assert buffer.getvalue() == ""
    assert "pulled depth-pro" in result.output


def test_pull_passes_no_callback_when_stderr_is_not_a_terminal(mocker, stderr):
    stderr(terminal=False)
    client = FakeClient()
    spy = mocker.spy(client, "models_pull")
    mocker.patch("splat.cli.models.get_client", return_value=client)

    runner.invoke(app, ["models", "pull", "depth-pro"])

    assert spy.call_args.kwargs["on_progress"] is None


def test_pull_prints_the_repo_and_size_on_a_terminal(mocker, stderr):
    buffer = stderr(terminal=True)
    mocker.patch("splat.cli.models.get_client", return_value=FakeClient())

    runner.invoke(app, ["models", "pull", "depth-pro"])

    assert "org/model" in buffer.getvalue()


def test_pull_failure_exits_non_zero_with_one_line(mocker, stderr):
    stderr(terminal=True)
    mocker.patch(
        "splat.cli.models.get_client", return_value=FakeClient(SplatDomainError("no such model"))
    )

    result = runner.invoke(app, ["models", "pull", "depth-pro"])

    assert result.exit_code == 1
