import json
import re
from pathlib import Path

from typer.testing import CliRunner

from splat.adapters.cache.filesystem import FilesystemAssetCache
from splat.cli.main import app
from splat.domain.asset import AssetKind
from splat.domain.value_objects import MIT
from tests.image_helpers import write_sample_png

runner = CliRunner()


class FakeCaptionBackend:
    name = "fake-captioner"
    license = MIT

    def caption(
        self,
        image_path: Path,
        *,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **params,
    ) -> str:
        return f"{prompt}: small scene"


def _sample_image(tmp_path: Path) -> Path:
    return write_sample_png(tmp_path / "scene.png", (3, 2))


def test_caption_file_path_input_ndjson_output(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.caption.get_caption_backend", return_value=FakeCaptionBackend())

    result = runner.invoke(
        app,
        ["caption", str(_sample_image(tmp_path)), "--model", "fake-captioner", "--prompt", "Look"],
    )

    assert result.exit_code == 0, result.output
    line = json.loads(result.output.strip().splitlines()[-1])
    assert line["kind"] == "caption"
    assert line["metadata"]["prompt"] == "Look"
    assert line["created_by"] == "caption:fake-captioner"


def test_caption_human_output(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.caption.get_caption_backend", return_value=FakeCaptionBackend())
    mocker.patch("splat.cli._pipeline_io.is_piped", return_value=False)

    result = runner.invoke(app, ["caption", str(_sample_image(tmp_path)), "--prompt", "Look"])
    plain_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)

    assert result.exit_code == 0, result.output
    assert "caption" in plain_output
    assert "Look: small scene" in plain_output


def test_caption_output_flag_writes_text(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.caption.get_caption_backend", return_value=FakeCaptionBackend())
    out_path = tmp_path / "caption.txt"

    result = runner.invoke(
        app, ["caption", str(_sample_image(tmp_path)), "-o", str(out_path), "--prompt", "Look"]
    )

    assert result.exit_code == 0, result.output
    assert out_path.read_text(encoding="utf-8") == "Look: small scene"


def test_caption_asset_id_input(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.caption.get_caption_backend", return_value=FakeCaptionBackend())
    cache = FilesystemAssetCache(tmp_path / "cache")
    asset = cache.put_external(_sample_image(tmp_path), kind=AssetKind.IMAGE)

    result = runner.invoke(app, ["caption", f"@{asset.id}"])

    assert result.exit_code == 0, result.output
    line = json.loads(result.output.strip().splitlines()[-1])
    assert line["parent_ids"] == [asset.id]


def test_caption_stdin_ndjson_input(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.caption.get_caption_backend", return_value=FakeCaptionBackend())
    cache = FilesystemAssetCache(tmp_path / "cache")
    asset = cache.put_external(_sample_image(tmp_path), kind=AssetKind.IMAGE)
    stdin_payload = json.dumps({"id": asset.id}) + "\n"

    result = runner.invoke(app, ["caption", "-"], input=stdin_payload)

    assert result.exit_code == 0, result.output
    line = json.loads(result.output.strip().splitlines()[-1])
    assert line["parent_ids"] == [asset.id]


def test_caption_unknown_model_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))

    result = runner.invoke(app, ["caption", str(_sample_image(tmp_path)), "--model", "nope"])

    assert result.exit_code == 1
    assert "Unknown caption model" in result.output
