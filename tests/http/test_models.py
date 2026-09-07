from pathlib import Path

from fastapi.testclient import TestClient

from splat.http.app import app

client = TestClient(app)


class FakeModelSource:
    def __init__(self) -> None:
        self.pulled: list[str] = []
        self.removed: list[str] = []

    def pull(self, model_id: str, *, revision: str | None = None) -> Path:
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
    assert {"name", "runtime", "license", "cached"} <= rows[0].keys()


def test_pull_unknown_model_returns_422(mocker):
    mocker.patch("splat.handlers.models.get_model_source", return_value=FakeModelSource())

    response = client.post("/models/not-a-real-model/pull")

    assert response.status_code == 422


def test_rm_delegates_to_model_source(mocker):
    fake = FakeModelSource()
    mocker.patch("splat.handlers.models.get_model_source", return_value=fake)

    response = client.delete("/models/mvsplat")

    assert response.status_code == 200
    assert len(fake.removed) == 1
