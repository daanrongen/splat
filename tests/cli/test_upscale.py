import json
import re
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from splat.adapters.cache.filesystem import FilesystemManifestRepository
from splat.adapters.formats.image import read_rgb_or_rgba
from splat.cli.main import app
from splat.domain.manifest import ManifestKind
from splat.domain.value_objects import MIT
from tests.image_helpers import write_sample_png

runner = CliRunner()


class FakeUpscaleBackend:
    name = "fake-upscaler"
    license = MIT
    supported_factors = (2, 4)

    def upscale(self, image, *, factor: int, tile: int = 0, **params):
        return np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)


def _patch_backend(mocker):
    return mocker.patch(
        "splat.handlers.upscale.get_upscale_backend", return_value=FakeUpscaleBackend()
    )


def _sample_image(tmp_path: Path) -> Path:
    return write_sample_png(tmp_path / "scene.png", (3, 2))


def test_upscale_file_path_input_ndjson_output(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    _patch_backend(mocker)

    result = runner.invoke(
        app,
        [
            "upscale",
            str(_sample_image(tmp_path)),
            "--model",
            "fake-upscaler",
            "--factor",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    line = json.loads(result.output.strip().splitlines()[-1])
    assert line["kind"] == "image"
    assert line["created_by"] == "upscale:fake-upscaler"
    assert line["metadata"]["output_width"] == 6
    assert line["metadata"]["output_height"] == 4


def test_upscale_human_output(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    _patch_backend(mocker)
    mocker.patch("splat.cli._pipeline_io.is_piped", return_value=False)

    result = runner.invoke(
        app,
        ["upscale", str(_sample_image(tmp_path)), "--model", "fake-upscaler"],
    )
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)

    assert result.exit_code == 0, result.output
    assert "upscaled" in plain
    assert "3x2 -> 12x8" in plain


def test_upscale_output_flag_writes_png(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    _patch_backend(mocker)
    out_path = tmp_path / "upscaled.png"

    result = runner.invoke(
        app,
        [
            "upscale",
            str(_sample_image(tmp_path)),
            "--model",
            "fake-upscaler",
            "-o",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert read_rgb_or_rgba(out_path).shape == (8, 12, 3)


def test_upscale_asset_id_input(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    _patch_backend(mocker)
    cache = FilesystemManifestRepository(tmp_path / "cache")
    asset = cache.put_external(_sample_image(tmp_path), kind=ManifestKind.IMAGE)

    result = runner.invoke(app, ["upscale", f"@{asset.id}", "--model", "fake-upscaler"])

    assert result.exit_code == 0, result.output
    line = json.loads(result.output.strip().splitlines()[-1])
    assert line["parent_ids"] == [asset.id]


def test_upscale_stdin_ndjson_input(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    _patch_backend(mocker)
    cache = FilesystemManifestRepository(tmp_path / "cache")
    asset = cache.put_external(_sample_image(tmp_path), kind=ManifestKind.IMAGE)

    result = runner.invoke(
        app,
        ["upscale", "-", "--model", "fake-upscaler", "--factor", "2"],
        input=json.dumps({"id": asset.id}) + "\n",
    )

    assert result.exit_code == 0, result.output
    line = json.loads(result.output.strip().splitlines()[-1])
    assert line["parent_ids"] == [asset.id]


def test_upscale_rejects_invalid_factor(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    _patch_backend(mocker)

    result = runner.invoke(
        app,
        [
            "upscale",
            str(_sample_image(tmp_path)),
            "--model",
            "fake-upscaler",
            "--factor",
            "3",
        ],
    )

    assert result.exit_code == 1
    assert "--factor 2, 4" in result.output
