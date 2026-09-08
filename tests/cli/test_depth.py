import json
import re
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from splat.adapters.cache.filesystem import FilesystemManifestRepository
from splat.adapters.formats.image import read_rgb
from splat.cli.main import app
from splat.domain.image_space import DepthMap
from splat.domain.manifest import ManifestKind
from splat.domain.value_objects import APPLE_ASCL
from tests.image_helpers import write_sample_png

runner = CliRunner()


class FakeDepthBackend:
    name = "depth-pro"
    license = APPLE_ASCL

    def estimate(self, image_path, **params) -> DepthMap:
        depth = np.arange(6, dtype=np.float32).reshape(2, 3)
        return DepthMap(
            depth=depth,
            focal_length_px=1234.5,
            field_of_view_deg=30.0,
            metadata={"source_model": "fake/depth", "device": "cpu"},
        )


def _sample_image(tmp_path: Path) -> Path:
    path = tmp_path / "scene.png"
    return write_sample_png(path, (3, 2))


def test_depth_file_path_input_ndjson_output(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.depth.get_depth_backend", return_value=FakeDepthBackend())
    image_path = _sample_image(tmp_path)

    result = runner.invoke(app, ["depth", str(image_path)])

    assert result.exit_code == 0, result.output
    line = json.loads(result.output.strip().splitlines()[-1])
    assert line["kind"] == "depth_map"
    assert line["metadata"]["focal_length_px"] == 1234.5
    assert line["created_by"] == "depth:depth-pro"


def test_depth_human_output(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.depth.get_depth_backend", return_value=FakeDepthBackend())
    mocker.patch("splat.cli._pipeline_io.is_piped", return_value=False)
    image_path = _sample_image(tmp_path)

    result = runner.invoke(app, ["depth", str(image_path)])
    plain_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)

    assert result.exit_code == 0, result.output
    assert "depth" in plain_output
    assert "focal_length=1234.5" in plain_output


def test_depth_output_flag_writes_normalized_png(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.depth.get_depth_backend", return_value=FakeDepthBackend())
    image_path = _sample_image(tmp_path)
    out_path = tmp_path / "depth.png"

    result = runner.invoke(app, ["depth", str(image_path), "-o", str(out_path)])

    assert result.exit_code == 0, result.output
    assert out_path.exists()
    saved = read_rgb(out_path)
    assert saved.min() == 0
    assert saved.max() == 255


def test_depth_unknown_model_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    image_path = _sample_image(tmp_path)

    result = runner.invoke(app, ["depth", str(image_path), "--model", "nope"])

    assert result.exit_code == 1
    assert "Unknown depth model" in result.output


def test_depth_asset_id_input(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.depth.get_depth_backend", return_value=FakeDepthBackend())
    cache = FilesystemManifestRepository(tmp_path / "cache")
    asset = cache.put_external(_sample_image(tmp_path), kind=ManifestKind.IMAGE)

    result = runner.invoke(app, ["depth", f"@{asset.id}"])

    assert result.exit_code == 0, result.output
    line = json.loads(result.output.strip().splitlines()[-1])
    assert line["parent_ids"] == [asset.id]


def test_depth_stdin_ndjson_input(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.depth.get_depth_backend", return_value=FakeDepthBackend())
    cache = FilesystemManifestRepository(tmp_path / "cache")
    asset = cache.put_external(_sample_image(tmp_path), kind=ManifestKind.IMAGE)
    stdin_payload = json.dumps({"id": asset.id}) + "\n"

    result = runner.invoke(app, ["depth", "-"], input=stdin_payload)

    assert result.exit_code == 0, result.output
    line = json.loads(result.output.strip().splitlines()[-1])
    assert line["parent_ids"] == [asset.id]
