import json
from pathlib import Path

import pytest

from splat.adapters.formats.ply import PlyWriter
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


def test_info_reports_point_count(tmp_path, synthetic_cloud, call_tool):
    ply_path = tmp_path / "a.ply"
    PlyWriter().write(synthetic_cloud, ply_path)

    result = call_tool("info", path=str(ply_path))

    assert result.is_error is False
    body = json.loads(result.content[0].text)
    assert body["points"] == synthetic_cloud.point_count


def test_validate_reports_valid(tmp_path, synthetic_cloud, call_tool):
    ply_path = tmp_path / "a.ply"
    PlyWriter().write(synthetic_cloud, ply_path)

    result = call_tool("validate", path=str(ply_path))

    assert result.is_error is False
    body = json.loads(result.content[0].text)
    assert body["points"] == synthetic_cloud.point_count


def test_models_list_reports_cached_flag(mocker, call_tool):
    mocker.patch("splat.handlers.models.get_model_source", return_value=FakeModelSource())

    result = call_tool("models_list")

    assert result.is_error is False


def test_models_pull_unknown_model_raises(mocker):
    fake = FakeModelSource()
    mocker.patch("splat.handlers.models.get_model_source", return_value=fake)

    with pytest.raises(SplatDomainError):
        models_handler.pull("not-a-real-model")
