from pathlib import Path

from splat.domain.value_objects import CC_BY_NC_SA_4_0, MIT
from splat.handlers.diffuse import DiffuseRequest, handle
from tests.image_helpers import sample_png_bytes


class FakeDiffusionBackend:
    name = "fake-diffuser"

    def __init__(self, license) -> None:
        self.license = license

    def diffuse(self, prompt, *, output_path: Path, **params) -> Path:
        output_path.write_bytes(sample_png_bytes())
        return output_path


def test_handle_returns_asset_with_no_warning_for_commercial_model(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch(
        "splat.handlers.diffuse.get_diffusion_backend",
        return_value=FakeDiffusionBackend(MIT),
    )

    result = handle(DiffuseRequest(prompt="a fox", model="fake-diffuser"))

    assert result.license_warning is None
    assert result.asset.content_path.read_bytes() == sample_png_bytes()
    assert result.asset.created_by == "diffuse:fake-diffuser"
    assert result.asset.metadata.output_width == 2
    assert result.asset.metadata.output_height == 2
    assert result.asset.content_size == len(sample_png_bytes())
    assert result.asset.content_sha256
    assert result.asset.created_at


def test_handle_warns_for_non_commercial_model(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch(
        "splat.handlers.diffuse.get_diffusion_backend",
        return_value=FakeDiffusionBackend(CC_BY_NC_SA_4_0),
    )

    result = handle(DiffuseRequest(prompt="a fox", model="fake-diffuser"))

    assert result.license_warning is not None
    assert "fake-diffuser" in result.license_warning
