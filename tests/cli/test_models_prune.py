from typer.testing import CliRunner

from splat.application.models_admin import OrphanedWeights
from splat.cli.main import app

runner = CliRunner()


def _prune(mocker, orphans, *args) -> str:
    mocker.patch("splat.handlers.models.prune", return_value=orphans)
    return runner.invoke(app, ["models", "prune", *args]).output


def test_prune_reports_nothing_to_do_when_the_store_is_clean(mocker):
    assert "nothing to prune" in _prune(mocker, [])


def test_prune_labels_an_unmanaged_directory(mocker):
    output = _prune(mocker, [OrphanedWeights("mlx-sam", 358_000_000)])

    assert "unmanaged" in output and "mlx-sam" in output


def test_prune_labels_a_hub_copy(mocker):
    output = _prune(mocker, [OrphanedWeights("apple/Sharp", 2_600_000_000, legacy=True)])

    assert "hub copy" in output and "apple/Sharp" in output


def test_prune_without_yes_says_how_to_delete(mocker):
    assert "--yes" in _prune(mocker, [OrphanedWeights("mlx-sam", 1)])


def test_prune_with_yes_reports_what_it_reclaimed(mocker):
    assert "reclaimed" in _prune(mocker, [OrphanedWeights("mlx-sam", 1)], "--yes")
