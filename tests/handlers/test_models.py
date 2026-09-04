from pathlib import Path

import pytest

from splat.domain.errors import SplatDomainError
from splat.handlers import models as models_handler


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


def test_list_models_reports_cached_flag(mocker):
    fake = FakeModelSource()
    mocker.patch("splat.handlers.models.get_model_source", return_value=fake)

    rows = models_handler.list_models()

    assert len(rows) > 0
    assert all(isinstance(cached, bool) for _, cached in rows)


def test_pull_delegates_to_model_source(mocker):
    fake = FakeModelSource()
    mocker.patch("splat.handlers.models.get_model_source", return_value=fake)

    models_handler.pull("triposr")

    assert len(fake.pulled) == 1


def test_pull_model_family_delegates_each_source(mocker):
    fake = FakeModelSource()
    mocker.patch("splat.handlers.models.get_model_source", return_value=fake)

    models_handler.pull("realesrgan-mlx")

    assert fake.pulled == [
        "mlx-community/Real-ESRGAN-x2plus",
        "mlx-community/Real-ESRGAN-x4plus",
    ]


def test_pull_unknown_model_raises(mocker):
    fake = FakeModelSource()
    mocker.patch("splat.handlers.models.get_model_source", return_value=fake)

    with pytest.raises(SplatDomainError):
        models_handler.pull("not-a-real-model")


def test_rm_delegates_to_model_source(mocker):
    fake = FakeModelSource()
    mocker.patch("splat.handlers.models.get_model_source", return_value=fake)

    models_handler.rm("triposr")

    assert len(fake.removed) == 1


def test_info_reports_upscale_model_sources():
    descriptor = models_handler.info("realesrgan-mlx")

    assert descriptor.name == "realesrgan-mlx"
    assert "mlx-community/Real-ESRGAN-x4plus" in descriptor.hf_repo_id
