from pathlib import Path

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from splat.domain.value_objects import CC_BY_NC_SA_4_0, MIT
from tests.image_helpers import sample_png_bytes


class FakeDiffusionBackend:
    name = "fake-diffuser"

    def __init__(self, license) -> None:
        self.license = license

    def diffuse(self, prompt, *, output_path: Path, **params) -> Path:
        output_path.write_bytes(sample_png_bytes())
        return output_path


def test_diffuse_returns_image_content(mocker, tmp_path, monkeypatch, call_tool):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch(
        "splat.handlers.diffuse.get_diffusion_backend",
        return_value=FakeDiffusionBackend(MIT),
    )

    result = call_tool("diffuse", prompt="a fox", model="fake-diffuser")

    assert result.is_error is False
    kinds = [block.type for block in result.content]
    assert "text" in kinds
    assert "image" in kinds


def test_diffuse_reports_non_commercial_license_warning(mocker, tmp_path, monkeypatch, call_tool):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch(
        "splat.handlers.diffuse.get_diffusion_backend",
        return_value=FakeDiffusionBackend(CC_BY_NC_SA_4_0),
    )

    result = call_tool("diffuse", prompt="a fox", model="fake-diffuser")

    texts = [block.text for block in result.content if block.type == "text"]
    assert any("warning" in t for t in texts)


def test_diffuse_unknown_model_is_reported_as_tool_error(tmp_path, monkeypatch, call_tool):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))

    with pytest.raises(ToolError, match="Unknown diffusion model"):
        call_tool("diffuse", prompt="a fox", model="nope")


def test_diffuse_missing_image_reports_the_cause(tmp_path, monkeypatch, call_tool):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))

    with pytest.raises(ToolError, match="nonexistent"):
        call_tool("diffuse", prompt="a fox", image=str(tmp_path / "nonexistent.png"))
