from typer.testing import CliRunner

from splat.cli.main import app
from splat.ports.client import ModelSummary

runner = CliRunner()


class FakeClient:
    def __init__(self, rows) -> None:
        self.rows = rows

    def models_list(self):
        return self.rows


def _list(mocker, *rows) -> str:
    mocker.patch("splat.cli.models.get_client", return_value=FakeClient(rows))
    return runner.invoke(app, ["models", "list"]).output


def test_list_shows_the_size_of_an_installed_model(mocker):
    row = ModelSummary("sd15-coreml", "diffuse", "coreml", "OpenRAIL-M", True, True, 911_000_000)

    assert "868.8 MB" in _list(mocker, row)


def test_list_shows_a_dash_for_a_model_that_is_not_installed(mocker):
    row = ModelSummary(
        "sdxl-turbo-mlx", "diffuse", "mlx", "StabilityAI-NC-Community", False, False, 0
    )

    output = _list(mocker, row)

    assert "GB" not in output and "-" in output


def test_list_never_folds_a_long_model_name(mocker):
    name = "depth-anything-v2-coreml"
    row = ModelSummary(name, "depth", "coreml", "Apache-2.0", True, True, 48_000_000)

    assert name in _list(mocker, row)


def test_list_names_unmanaged_directories(mocker):
    row = ModelSummary("mlx-sam", "-", "-", "-", True, True, 358_000_000, managed=False)

    assert "mlx-sam" in _list(mocker, row)


def test_list_shows_a_dash_for_a_model_without_weights(mocker):
    row = ModelSummary("mlx3d-capture", "gaussian", "mlx", "MIT", True, True, 0)

    assert "0.0 B" not in _list(mocker, row)


def test_list_marks_an_installed_model(mocker):
    row = ModelSummary("sd15-coreml", "diffuse", "coreml", "OpenRAIL-M", True, True, 911_000_000)

    assert "installed" in _list(mocker, row)


def test_list_marks_a_model_that_is_not_installed(mocker):
    row = ModelSummary("sdxl-turbo-mlx", "diffuse", "mlx", "SAI", False, False, 0)

    assert "not installed" in _list(mocker, row)


def test_list_marks_a_model_that_needs_no_weights(mocker):
    row = ModelSummary("mlx3d-capture", "gaussian", "mlx", "MIT", True, True, 0)

    assert "no weights" in _list(mocker, row)


def test_list_marks_an_unmanaged_directory(mocker):
    row = ModelSummary("mlx-sam", "-", "-", "-", True, True, 358_000_000, managed=False)

    assert "unmanaged" in _list(mocker, row)
