from pathlib import Path

from fastapi.testclient import TestClient

from splat.domain.value_objects import CC_BY_NC_SA_4_0, MIT
from splat.http.app import app
from tests.image_helpers import sample_png_bytes

client = TestClient(app)


class FakeDiffusionBackend:
    name = "fake-diffuser"

    def __init__(self, license) -> None:
        self.license = license

    def diffuse(self, prompt, *, output_path: Path, **params) -> Path:
        output_path.write_bytes(sample_png_bytes())
        return output_path


def test_diffuse_returns_asset_bytes(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch(
        "splat.handlers.diffuse.get_diffusion_backend",
        return_value=FakeDiffusionBackend(MIT),
    )

    response = client.post("/diffuse", json={"prompt": "a fox", "model": "fake-diffuser"})

    assert response.status_code == 200, response.text
    assert response.content == sample_png_bytes()
    assert "X-Splat-Asset-Id" in response.headers
    assert "X-Splat-License-Warning" not in response.headers


def test_diffuse_reports_non_commercial_license_warning(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch(
        "splat.handlers.diffuse.get_diffusion_backend",
        return_value=FakeDiffusionBackend(CC_BY_NC_SA_4_0),
    )

    response = client.post("/diffuse", json={"prompt": "a fox", "model": "fake-diffuser"})

    assert response.status_code == 200, response.text
    assert "X-Splat-License-Warning" in response.headers


def test_diffuse_unknown_model_returns_422(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))

    response = client.post("/diffuse", json={"prompt": "a fox", "model": "nope"})

    assert response.status_code == 422
    assert "Unknown diffusion model" in response.json()["detail"]
