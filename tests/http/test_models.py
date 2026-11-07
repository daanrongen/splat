import asyncio
import json
import threading
from pathlib import Path

from fastapi.testclient import TestClient

from splat.domain.errors import SplatDomainError
from splat.http import models as models_route
from splat.http.app import app

client = TestClient(app)


class FakeModelSource:
    def __init__(self) -> None:
        self.pulled: list[str] = []
        self.removed: list[str] = []

    def pull(self, model_id: str, *, revision: str | None = None, on_progress=None) -> Path:
        self.pulled.append(model_id)
        return Path("/fake")

    def is_cached(self, model_id: str) -> bool:
        return model_id in self.pulled

    def local_path(self, model_id: str) -> Path | None:
        return None

    def remove(self, model_id: str) -> None:
        self.removed.append(model_id)

    def list_cached(self) -> list[str]:
        return self.pulled


def test_list_models_returns_json_array(mocker):
    mocker.patch("splat.handlers.models.get_model_source", return_value=FakeModelSource())

    response = client.get("/models")

    assert response.status_code == 200
    rows = response.json()
    assert len(rows) > 0
    assert {"name", "stage", "runtime", "license", "commercial", "cached"} <= rows[0].keys()


def test_pull_unknown_model_returns_422(mocker):
    mocker.patch("splat.handlers.models.get_model_source", return_value=FakeModelSource())

    response = client.post("/models/not-a-real-model/pull")

    assert response.status_code == 422


def test_rm_delegates_to_model_source(mocker):
    fake = FakeModelSource()
    mocker.patch("splat.handlers.models.get_model_source", return_value=fake)

    response = client.delete("/models/depth-pro")

    assert response.status_code == 200
    assert len(fake.removed) == 1


class ReportingSource(FakeModelSource):
    def pull(self, model_id: str, *, revision: str | None = None, on_progress=None) -> Path:
        on_progress(50, 100)
        on_progress(100, 100)
        return super().pull(model_id)


def _events(response) -> list[dict]:
    return [json.loads(line) for line in response.text.splitlines()]


def test_pull_streams_progress_then_pulled(mocker):
    mocker.patch("splat.handlers.models.get_model_source", return_value=ReportingSource())

    response = client.post("/models/depth-pro/pull")

    assert response.status_code == 200
    assert _events(response) == [
        {"repo": "apple/DepthPro-hf", "done": 50, "total": 100},
        {"repo": "apple/DepthPro-hf", "done": 100, "total": 100},
        {"pulled": "depth-pro"},
    ]


def test_pull_reports_a_domain_error_as_the_last_event(mocker):
    class FailingSource(FakeModelSource):
        def pull(self, model_id, *, revision=None, on_progress=None):
            raise SplatDomainError("repo gated")

    mocker.patch("splat.handlers.models.get_model_source", return_value=FailingSource())

    response = client.post("/models/depth-pro/pull")

    assert _events(response) == [{"code": "domain_error", "detail": "repo gated"}]


def test_pull_reports_an_unexpected_error_as_the_last_event(mocker):
    class BrokenSource(FakeModelSource):
        def pull(self, model_id, *, revision=None, on_progress=None):
            raise RuntimeError("boom")

    mocker.patch("splat.handlers.models.get_model_source", return_value=BrokenSource())

    response = client.post("/models/depth-pro/pull")

    assert _events(response) == [{"code": "internal_error", "detail": "RuntimeError: boom"}]


def test_closing_the_stream_stops_the_download(mocker):
    stopped = threading.Event()
    first_report = threading.Event()
    resume = threading.Event()

    class SlowSource(FakeModelSource):
        def pull(self, model_id, *, revision=None, on_progress=None):
            on_progress(1, 100)
            first_report.set()
            resume.wait(5)
            try:
                on_progress(2, 100)
            except Exception:
                stopped.set()
                raise

    mocker.patch("splat.handlers.models.get_model_source", return_value=SlowSource())

    async def disconnect_after_first_event():
        events = models_route._pull_events("depth-pro")
        await anext(events)
        await events.aclose()

    asyncio.run(disconnect_after_first_event())
    resume.set()

    assert stopped.wait(5)
