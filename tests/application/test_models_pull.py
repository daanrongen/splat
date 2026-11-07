from splat.application.models_admin import PullModelUseCase


class FakeModelSource:
    def __init__(self) -> None:
        self.pulled: list[str] = []

    def pull(self, model_id, *, revision=None, on_progress=None):
        self.pulled.append(model_id)
        if on_progress is not None:
            on_progress(5, 10)


def test_execute_pulls_every_repo_of_a_model():
    source = FakeModelSource()

    PullModelUseCase(source).execute("realesrgan-mlx")

    assert source.pulled == ["mlx-community/Real-ESRGAN-x2plus", "mlx-community/Real-ESRGAN-x4plus"]


def test_execute_tags_each_report_with_its_repo_id():
    reports = []

    PullModelUseCase(FakeModelSource()).execute(
        "realesrgan-mlx", lambda *report: reports.append(report)
    )

    assert reports == [
        ("mlx-community/Real-ESRGAN-x2plus", 5, 10),
        ("mlx-community/Real-ESRGAN-x4plus", 5, 10),
    ]


def test_execute_without_a_callback_asks_the_source_for_none():
    seen = []

    class Source(FakeModelSource):
        def pull(self, model_id, *, revision=None, on_progress=None):
            seen.append(on_progress)

    PullModelUseCase(Source()).execute("depth-pro")

    assert seen == [None]
