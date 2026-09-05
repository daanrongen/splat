import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from splat.adapters.formats.ply import PlyWriter
from splat.cli.main import app
from splat.domain.manifest import ManifestKind
from splat.image_io import encode_png

runner = CliRunner()

_FAKE_PNG = encode_png(np.zeros((4, 8, 3), dtype=np.uint8))


class FakeRenderBackend:
    name = "fake-render"

    def render(self, cloud, output_path, **params):
        output_path.write_bytes(_FAKE_PNG)


def test_blender_file_input_emits_ndjson_asset(mocker, tmp_path, monkeypatch, synthetic_cloud):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.blender.get_render_backend", return_value=FakeRenderBackend())

    ply_path = tmp_path / "cloud.ply"
    PlyWriter().write(synthetic_cloud, ply_path)

    result = runner.invoke(app, ["blender", str(ply_path)])

    assert result.exit_code == 0, result.output
    line = json.loads(result.output.strip().splitlines()[-1])
    assert line["kind"] == ManifestKind.IMAGE.value
    assert line["created_by"] == "blender:fake-render"


def test_blender_output_option_writes_file(mocker, tmp_path, monkeypatch, synthetic_cloud):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.blender.get_render_backend", return_value=FakeRenderBackend())

    ply_path = tmp_path / "cloud.ply"
    PlyWriter().write(synthetic_cloud, ply_path)
    output_path: Path = tmp_path / "out.png"

    result = runner.invoke(app, ["blender", str(ply_path), "-o", str(output_path)])

    assert result.exit_code == 0, result.output
    assert output_path.read_bytes() == _FAKE_PNG
