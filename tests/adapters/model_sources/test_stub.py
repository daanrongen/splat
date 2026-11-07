import pytest

from splat.adapters.model_sources.stub import StubModelSource


@pytest.fixture(autouse=True)
def stub_cache(tmp_path, monkeypatch, mocker):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("SPLAT_MODEL_CACHE_DIR", str(tmp_path / "models"))
    mocker.patch("splat.adapters.model_sources.stub.time.sleep")


def test_pull_reports_increasing_progress_up_to_the_total():
    reports = []

    StubModelSource().pull("some/model", on_progress=lambda *report: reports.append(report))

    done = [d for d, _ in reports]
    assert done == sorted(done)
    assert reports[-1][0] == reports[-1][1]


def test_pull_marks_the_model_cached():
    source = StubModelSource()

    source.pull("some/model")

    assert source.is_cached("some/model") is True


def test_pull_of_a_cached_model_reports_nothing():
    source = StubModelSource()
    source.pull("some/model")
    reports = []

    source.pull("some/model", on_progress=lambda *report: reports.append(report))

    assert reports == []


def test_remove_clears_the_cached_model():
    source = StubModelSource()
    source.pull("some/model")

    source.remove("some/model")

    assert source.is_cached("some/model") is False
